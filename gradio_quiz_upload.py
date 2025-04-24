import gradio as gr

import os
import shutil
import tempfile
from pdf2image import convert_from_path
import pytesseract
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma



# Directory setup
SUMMARY_STORE = "../plc_storage/data101_stored_prompts"
CHROMA_DB_DIR = "../plc_storage/data101_chroma_db"

os.makedirs(SUMMARY_STORE, exist_ok=True)

# Initialize OpenAI
openai_client = OpenAI(api_key="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")

# Initialize ChromaDB with HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)

# Extract text from a PDF file using pytesseract
def extract_text_from_pdf(file):
    if file is None:
        return ""
    filepath = file.name  # Gradio provides the path to the uploaded file
    images = convert_from_path(filepath)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def summarize_quiz(quiz_pdf):
    quiz_text = extract_text_from_pdf(quiz_pdf)
    if not quiz_text.strip():
        return "No content to summarize."
    print("DEBUG - QUIZ TEXT" + quiz_text)
    prompt = f"find 10 topics from this quiz: \n{quiz_text}"
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    summary = response.choices[0].message.content.strip()
    # print("DEBUG - SUMMARY" + summary)
    return summary

def submit_summary_and_material(summary_text, related_pdf, logistics_text):
    summary_id = "summary_text"

    with open(os.path.join(SUMMARY_STORE, f"{summary_id}.txt"), "w") as f:
        f.write(summary_text)

    if logistics_text.strip():
        with open(os.path.join(SUMMARY_STORE, "logistic_prompt.txt"), "w") as f:
            f.write(logistics_text)

    related_text = extract_text_from_pdf(related_pdf)
    if related_text.strip():
        vectordb.add_texts([related_text], ids=[summary_id])

    return f"Summary and related material stored with ID: {summary_id}"

def load_and_store_uploaded_pdf(filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    write_to_vectordb(pages)

def write_to_vectordb(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(pages)
    persist_directory = "../plc_storage/data101_chroma_db"

    global vectordb
    if vectordb is not None:
        vectordb = None  # Dereference

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(vectordb._collection.count())

with gr.Blocks() as demo:
    with gr.Row():
        quiz_upload = gr.File(label="Upload Quiz (PDF)", file_types=[".pdf"])
        related_upload = gr.File(label="Upload Related Material (PDF)", file_types=[".pdf"])
    
    logistics_box = gr.Textbox(label="Exam Logistics", lines=3)
    summarize_btn = gr.Button("Summarize Quiz")
    summary_box = gr.Textbox(label="Editable Summary", lines=10)

    

    submit_btn = gr.Button("Submit Summary and Material")
    status = gr.Textbox(label="Status", interactive=False)

    summarize_btn.click(fn=summarize_quiz, inputs=quiz_upload, outputs=summary_box)
    submit_btn.click(fn=submit_summary_and_material, inputs=[summary_box, related_upload, logistics_box], outputs=status)

demo.launch(server_name="plc.cs.rutgers.edu", server_port=443, ssl_keyfile="/etc/letsencrypt/live/plc.cs.rutgers.edu/privkey.pem", ssl_certfile="/etc/letsencrypt/live/plc.cs.rutgers.edu/fullchain.pem")
# demo.launch(share=True)