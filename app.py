# Import necessary libraries
import os
from flask import Flask, request, jsonify, render_template
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from datetime import datetime
import shutil

# Create Flask application instance
app = Flask(__name__)

# Print a startup message
print("[STARTUP] Initializing Flask application...")

# Load environment variables from a .env file
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizers parallelism warning

# Configuration settings
DATABASE_FILE = "static/assets/database.txt"  # Path to database file
TODO_FILE = "static/assets/todo_list.txt"  # File for storing persistent To-Do list
CHROMA_PATH = "chroma"  # Path for Chroma DB
COLLECTION_NAME = "my_collection"  # Collection name for Chroma DB
GROQ_API_KEY = os.environ['GROQ_API_KEY']  # API key for Groq LLM

# Print configuration information
print(f"[CONFIG] Using Chroma path: {CHROMA_PATH}, Collection: {COLLECTION_NAME}")

# Ensure that necessary directories and files exist
os.makedirs("static/assets", exist_ok=True)
if not os.path.exists(DATABASE_FILE):
    open(DATABASE_FILE, "w").close()
if not os.path.exists(TODO_FILE):
    open(TODO_FILE, "w").close()

# Function to clear and reinitialize the Chroma database
def clear_chroma_db(path):
    """Clears the Chroma database directory."""
    print("[INIT] Clearing old Chroma DB...")
    if os.path.exists(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Delete files
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove non-empty directories
            except Exception as e:
                print(f"[ERROR] Failed to clear file {file}: {str(e)}")
    os.makedirs(path, exist_ok=True)

# Function to initialize HuggingFace embeddings
def initialize_huggingface_embeddings():
    """Initializes HuggingFace embeddings."""
    print("[INIT] Initializing HuggingFace embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("[INIT] HuggingFace embeddings initialized successfully")
        return embeddings
    except Exception as e:
        print(f"[ERROR] Failed to initialize HuggingFace embeddings: {str(e)}")
        raise

# Function to initialize LangChain Groq LLM
def initialize_groq_llm():
    """Initializes Groq LLM."""
    print("[INIT] Initializing Groq LLM...")
    try:
        llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.0,
            max_tokens=None,
            timeout=1000,
            max_retries=2
        )
        print("[INIT] Groq LLM initialized successfully")
        return llm
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq LLM: {str(e)}")
        raise

# Function to initialize Chroma vector store
def initialize_chroma_vectorstore(embedding_function):
    """Initializes Chroma vector store."""
    print("[INIT] Initializing Chroma vector store...")
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH,
        )
        # Populate vectorstore with data from the database file
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, "r") as file:
                lines = file.readlines()
                if lines:
                    print("[INIT] Loading data from database.txt into Chroma vector store...")
                    vectorstore.add_texts(
                        texts=[line.strip() for line in lines],
                        metadatas=[{"source": "database_file"}] * len(lines),
                        ids=[f"line_{i}" for i in range(len(lines))]
                    )
        print("[INIT] Chroma vector store initialized successfully")
        return vectorstore
    except Exception as e:
        print(f"[ERROR] Failed to initialize Chroma vector store: {str(e)}")
        raise

# Function to set up the QA chain
def setup_qa_chain(llm, retriever):
    """Sets up the QA chain using the given LLM and retriever."""
    print("[INIT] Setting up QA chain...")
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )
        print("[INIT] QA chain setup complete")
        return qa_chain
    except Exception as e:
        print(f"[ERROR] Failed to setup QA chain: {str(e)}")
        raise

# Function to add a task to the To-Do List file
def add_to_todo_file(task):
    """Add a task with a timestamp to the To-Do List file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{task} | {timestamp}"
    with open(TODO_FILE, "a") as file:
        file.write(entry + "\n")
    print(f"[TODO FILE] Task added: {entry}")
    # Also add the task to the database file
    with open(DATABASE_FILE, "a") as db_file:
        db_file.write(task + "\n")
    print(f"[DATABASE FILE] Task added to database: {task}")
    # Add the task to the vector store
    vectorstore.add_texts(
        texts=[task],
        metadatas=[{"source": "todo_task"}],
        ids=[f"task_{datetime.now().timestamp()}"]
    )
    print(f"[VECTORSTORE] Task added to vector store: {task}")

# Function to save the To-Do List file with the latest tasks
def save_todo_file(tasks):
    """Overwrite the To-Do List file with the latest tasks."""
    with open(TODO_FILE, "w") as file:
        for task in tasks:
            file.write(f"{task['task']} | {task['timestamp']}\n")
    print("[TODO FILE] Tasks saved successfully.")

# Function to load all tasks from the To-Do List file
def load_todo_file():
    """Load all tasks from the To-Do List file with their timestamps."""
    tasks = []
    with open(TODO_FILE, "r") as file:
        for line in file.readlines():
            if " | " in line:
                task, timestamp = line.strip().split(" | ", 1)
                tasks.append({"task": task, "timestamp": timestamp})
            else:
                tasks.append({"task": line.strip(), "timestamp": "Unknown"})
    return tasks

# Function to detect whether a task is related to a future event
def detect_future_task(text, qa_chain):
    """Determine if a task involves a future event."""
    print("[PROCESS] Detecting future tasks...")
    prompt = f"""
You are an assistant that determines whether a task contains a future time reference (e.g., tomorrow, next week, specific dates).
If the input task involves a future time reference, respond with "Future-related task detected."
If it does not, respond with "No future task detected."

Input:
{text}

Output:
"""
    response = qa_chain.invoke({"query": prompt})
    result = response.get("result", "No future task detected.")
    print(f"[TASK DETECTION] Result: {result}")
    return "Future-related task detected." in result

# Initialize components
clear_chroma_db(CHROMA_PATH)
embedding_function = initialize_huggingface_embeddings()
groq_llm = initialize_groq_llm()
vectorstore = initialize_chroma_vectorstore(embedding_function)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = setup_qa_chain(groq_llm, retriever)

# Flask route to serve the index page
@app.route("/")
def index():
    print("[REQUEST] Serving index page")
    todo_tasks = load_todo_file()
    return render_template("index.html", todo_tasks=todo_tasks)

# Flask route for chatbot requests
@app.route("/chatbot/", methods=["POST"])
def chatbot():
    print("\n[REQUEST] New chatbot request received")
    try:
        data = request.get_json()
        message = data.get("message", "").strip().lower()
        print(f"[INPUT] Message: {message}")

        todo_task = None
        final_response = ""

        # If message starts with "question", process it as a question
        if message.startswith("question"):
            print("[PROCESS] Detected QUERY operation")
            query_message = message.replace("question:", "").strip()
            query_with_prompt = f"""
You are an assistant designed to answer questions using a personal database. If the information needed to answer a question is not available in the database, respond with "I don't have that information right now."

{query_message}
"""
            response = qa_chain.invoke({"query": query_with_prompt})
            final_response = response.get("result", "An error occurred.")

        # If message starts with "add", add the task to the To-Do List or database
        elif message.startswith("add"):
            print("[PROCESS] Detected ADD operation")
            add_message = message.replace("add:", "").strip()
            add_message = add_message.replace("add ", "").strip()

            if detect_future_task(add_message, qa_chain):
                add_to_todo_file(add_message)
                todo_task = add_message
                final_response = "Task added to To-Do List and database."
            else:
                with open(DATABASE_FILE, "a") as file:
                    file.write(add_message + "\n")
                vectorstore.add_texts(
                    texts=[add_message],
                    metadatas=[{"source": "database_file"}],
                    ids=[f"task_{datetime.now().timestamp()}"]
                )
                final_response = "Task stored in database and vector store."

        # Handle other cases where the message does not match the expected format
        else:
            final_response = "Please start your input with 'question:' or 'add:'."

        return jsonify({"response": final_response, "todo_task": todo_task})
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return jsonify({"response": error_message, "todo_task": None})

# Flask route to add a task to the To-Do List
def add_todo():
    try:
        data = request.get_json()
        task = data.get("task", "").strip()
        if not task:
            return jsonify({"status": "error", "message": "Invalid task"}), 400
        add_to_todo_file(task)
        return jsonify({"status": "success", "task": task})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Flask route to delete a task from the To-Do List
@app.route("/delete_todo/", methods=["POST"])
def delete_todo():
    try:
        data = request.get_json()
        task = data.get("task", "").strip()
        todo_tasks = load_todo_file()
        matching_tasks = [t for t in todo_tasks if t["task"] == task]
        if matching_tasks:
            todo_tasks.remove(matching_tasks[0])
            save_todo_file(todo_tasks)
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Task not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Entry point for running the Flask application
if __name__ == "__main__":
    print("[STARTUP] Starting Flask server...")
    app.run(debug=False, host="0.0.0.0", port=8081)
