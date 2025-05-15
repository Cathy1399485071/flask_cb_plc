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
openai_client = OpenAI(api_key="OPENAI_API_KEY", model="gpt-3.5-turbo")

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
    # print("DEBUG - QUIZ TEXT" + quiz_text)
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
    
    load_and_store_uploaded_pdf(related_pdf)
    return f"Summary and related material stored with ID: {summary_id}"

def write_to_vectordb(pages):
    # Check if there are pages to process
    if not pages or len(pages) == 0:
        print("No pages to process. Skipping vector DB insertion.")
        return

    try:
        # Split pages into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_documents(pages)
        print(f"Got {len(splits)} chunks after splitting.")
    except Exception as e:
        print(f"Error during text splitting: {e}")
        return

    if len(splits) == 0:
        print("No content to insert into ChromaDB after splitting. Skipping.")
        return

    persist_directory = "../plc_storage/data101_chroma_db"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        # If ChromaDB already exists, load it
        if os.path.exists(persist_directory):
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model
            )
            print("Loaded existing ChromaDB.")
        else:
            # Otherwise, create a new database with the documents
            vectordb = Chroma.from_documents(
                documents=splits,
                embedding=embedding_model,
                persist_directory=persist_directory
            )
            print("Created new ChromaDB and inserted documents.")
    except Exception as e:
        print(f"Error loading or creating ChromaDB: {e}")
        return

    try:
        vectordb.persist()
        print("ChromaDB write complete.")
    except Exception as e:
        print(f"Error persisting ChromaDB: {e}")

def load_and_store_uploaded_pdf(filepath):
    # Check if filepath is valid
    if not filepath:
        print("No file path provided. Skipping PDF loading.")
        return

    try:
        # Try loading the PDF
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from PDF.")
    except Exception as e:
        print(f"Error loading PDF file: {e}")
        return

    # If no pages were loaded, stop
    if not pages or len(pages) == 0:
        print("No pages found in the PDF. Skipping processing.")
        return

    # Preview sample pages
    for i, page in enumerate(pages[:3]):
        print(f"Page {i+1} content sample: {page.page_content[:100]}")

    # Pass the pages to vector database storage
    write_to_vectordb(pages)


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