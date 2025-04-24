from flask import Flask, render_template, request

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain.chains import RetrievalQA
import os
from langchain_community.chat_models import ChatOpenAI
import csv
from datetime import datetime
from module.config import DS_CHROMA_DB_DIR, DATA101_CHROMA_DB_DIR
from module.config import DATASCI, DATA101
from module.config import OPENAI_API_KEY
from module.config import DS_SUMMARY_STORE, DATA101_SUMMARY_STORE

is_initilized = False


def get_chatbot_response(user_input, selected_course, username):
    global is_initilized, llm, rqa, course_summary_path
    global content_summary, logistic_summary
    if selected_course is None:
        return "Error: Invalid course selection"
    if not is_initilized:
        print("DEBUG - initializing the chatbot")
        llm, rqa, course_summary_path = initialize(selected_course)
        content_summary, logistic_summary = get_custom_prompt_template(course_summary_path)
        is_initilized = True
    if not user_input:
        return "Error: No query provided"
    history_filepath = "../plc_storage/history/" + username + "_history.csv"
    query_answer_history = get_last_three_chats(history_filepath)
    print("DEBUG - query_answer_history: ", query_answer_history)
    result = rqa({
            "query": user_input,
            "topics_in_quiz": f"""The quiz will cover the following topics:\n{content_summary}\n\nBased on this, answer the question\n{user_input}, if you don't know, say I don't know."""
        })
    response = result["result"]
    fallback_phrases = [
        "text doesn't provide",
        "text does not provide",
        "information is not available",
        "based on the provided context",
        "not mentioned in the context",
        "I do not have access to",
        "text doesn't contain",
        "I'm sorry",
        "don't know"
    ]
    if any(phrase.lower() in response.lower() for phrase in fallback_phrases):
        category = categorize_query(user_input, llm)
        if category is None:
            return "Error: Unable to categorize query"
        prompt = set_prompt(content_summary, logistic_summary, user_input, category, query_answer_history)
        response = get_response_from_llm(llm, prompt)
    if response is None:
        return "Error: Unable to get response from LLM"
    log_to_csv(user_input, response, history_filepath)
    return response


def initialize(class_name):
    # Initialize the LLM with OpenAI API key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
    
    # Initialize the vector store with Chroma
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if(class_name == DATASCI):
        directory = DS_CHROMA_DB_DIR
        course_summary_path = DS_SUMMARY_STORE
    else:
        directory = DATA101_CHROMA_DB_DIR
        course_summary_path = DATA101_SUMMARY_STORE
    vectordb = Chroma(persist_directory=directory, embedding_function=embedding_model)
    retriever = vectordb.as_retriever()
    retriverQA = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )
    
    return llm, retriverQA, course_summary_path


def get_custom_prompt_template(course_summary_path):
    # Load the custom prompt template from a file
    with open(course_summary_path + "/summary_text.txt", "r", encoding="utf-8") as file:
        custom_prompt_template = file.read()
    
    # Load the logistic prompt template from a file
    with open(course_summary_path + "/logistic_prompt.txt", "r", encoding="utf-8") as file:
        custom_logistic_prompt_template = file.read()
    
    return custom_prompt_template, custom_logistic_prompt_template


def categorize_query(query_text, llm):
    if not query_text:
        return "Error: No query provided"
    
    categorize_prompt = f"""
    You are a classifier that assigns a user query into one of four categories based on its content and intent. The categories are:  
    a. logistic - questions about schedule, requirements, deadlines, availability, or course policies.  
    b. conceptual - questions seeking to understand or clarify a theory, definition, or abstract idea.  
    c. scenario - questions that present a hypothetical or real-world situation and ask how a concept applies in that context.  
    d. coding - questions involving code writing, debugging, syntax issues, or implementation.
    e. what on quiz - ask for what topics are on the quiz

    Task: Given a query, respond with only one category name: logistic, conceptual, scenario, or coding.

    Query: "{query_text}"  
    Category:"""
    query_category = llm.invoke(categorize_prompt)
    category = query_category.content.lower()
    
    return category


def set_prompt(content_summary, logistic_summary, query_text, category, query_answer_history):
    if "logistic" in category:
        # print("DEBUG - this is identified as a logistic question")
        query_answer_prompt = f"""Answer the logistic information based on the exact information:\n{logistic_summary}\nAnswer the question:\n{query_text}\nIf you don't know, just say you don't know."""
    elif "conceptual" in category:
        # print("DEBUG - this is identified as a conceptual question")
        query_answer_prompt = f"""
        You are an assistant helping students review for a quiz.

        You will receive:
        1. A summary of key topics and terminology likely to appear on the quiz.
        2. The student's last few questions and your previous answers (for reference only).

        Use the summary to answer the new question clearly and concisely.  
        If the previous chats are relevant to the new query, take them into account — otherwise, ignore them.

        Quiz Topics Summary:
        {content_summary}

        Recent Chat History (optional context):
        {query_answer_history}

        Now answer the following conceptual question using appropriate terminology or concepts from the quiz summary above.

        - Explain clearly and concisely in **80 words or less**.

        Query:  
        "{query_text}"

        Answer:
        """
    elif "coding" in category:
        # print("DEBUG - this is identified as a coding question")
        query_answer_prompt = f"""
        You are an assistant helping students review for a quiz.

        You will receive:
        1. A summary of relevant methods and topics likely to appear on the quiz.
        2. The student's recent questions and your past answers (provided only for context — use them only if relevant).

        Use the quiz summary to solve the coding problem.  
        If the previous chats are helpful to solving the current question, consider them. Otherwise, ignore them.

        Quiz Topics Summary:
        {content_summary}

        Recent Chat History (optional context):
        {query_answer_history}

        When given a coding question, treat it as a problem to solve using the **methods or techniques mentioned above**.  
        - If relevant methods are mentioned, write your solution using them.  
        - If no methods are provided in the summary, you may choose any reasonable approach.

        Then, provide a **brief explanation** of your code in 1-2 sentences.

        Query:  
        "{query_text}"

        Answer:
        """
    elif "scenario" in category:
        # print("DEBUG - this is identified as a step scenario query")
        query_answer_prompt = f"""
        You are an assistant helping students review for a quiz.

        You are given:
        1. A summary of the approaches and methods relevant to the quiz.
        2. The student's recent questions and your past answers (for context — use only if they help with the current question).

        Quiz Topics Summary:
        {content_summary}

        Recent Chat History (optional context):
        {query_answer_history}

        Task:
        - First, try to answer the question using the relevant approaches or methods from the summary above.
        - If none of those methods apply to the question, answer the question in the most helpful and clear way you can.
        - Do not justify whether the question is related to the quiz or not — just provide the best explanation possible.
        - Keep your explanation focused and sequential, with just enough detail to support quiz-level understanding.

        Query:  
        "{query_text}"

        Answer:
        """
    elif "what on quiz" in category:
        # print("DEBUG - this is identified as a what on quiz question")
        query_answer_prompt = f"""
        The quiz will cover the following topics:
        {content_summary}

        Below is the student's recent chat history for optional context.  
        Use it only if you find a meaningful connection to the current query.

        Recent Chat History:
        {query_answer_history}

        Use the terminology, methods, or concepts listed above to answer the following query as clearly and concisely as possible.  
        If none of the listed topics apply, still provide the best helpful answer you can.

        Query: {query_text}  
        Answer (max 50 words):
        """
    else:
        # print("DEBUG - not categories")
        query_answer_prompt = f"""
        The quiz will cover the following topics:
        {content_summary}

        You will also be shown the student's recent queries and your answers (for reference only).  
        Use them **only if they help clarify or connect to the current query**.

        Recent Chat History (optional context):
        {query_answer_history}

        Use the terminology, methods, or concepts listed above to answer the following query as clearly and concisely as possible.  
        If none of the listed topics apply, still provide the best helpful answer you can.

        Query: {query_text}  
        Answer (max 50 words):
        """
    return query_answer_prompt


def get_response_from_llm(llm, query_answer_prompt):
    # Get the response from the LLM
    response = llm.invoke(query_answer_prompt)
    response_content = response.content
    if "logistic" in query_answer_prompt.lower():
        response_content = response_content + " For more information about logistic, please refer to the canvas announcement or reach out to course TA."
    return response_content


def log_to_csv(query, response, filepath):
    file_exists = os.path.exists(filepath)

    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Query", "Response"])  # write header only once
        writer.writerow([datetime.now(), query, response])

def get_last_three_chats(filepath):
    if not os.path.exists(filepath):
        return ""

    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = list(csv.reader(file))
        rows = reader[1:]  # Skip header if exists

    if not rows:
        return ""

    last_chats = rows[-3:]  # Get up to last 3 rows
    formatted = ""
    for timestamp, query, response in last_chats:
        formatted += f"[{timestamp}]\nUser: {query}\nBot: {response}\n\n"

    return formatted.strip()