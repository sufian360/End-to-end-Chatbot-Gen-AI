## Backend (`app.py`)



from flask import Flask, render_template, request, redirect, url_for, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load environment variables
load_dotenv()

# Set up API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Initialize Embeddings
embeddings = download_hugging_face_embeddings()


index_name = "chatbot"
#  Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.4, "max_length": 500},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Create Chains
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
# In-memory user database (temporary, for demo purposes)
import traceback

users = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users:
            return "Email already registered. Please log in."
        users[email] = password
        session['user'] = email
        return redirect(url_for('chat'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if users.get(email) == password:
            session['user'] = email
            return redirect(url_for('chat'))
        else:
            return "Invalid email or password."
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/chat')
def chat():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    if 'user' not in session:
        return "Unauthorized. Please login."
    
    msg = request.form.get("msg")
    if not msg:
        return "Please provide a message."
    
    try:
        if msg.lower() in ["hi", "hello", "hey", "how are you"]:
            return "Hello! How can I assist you today?"
        
        print(f"User query: {msg}")
        response = rag_chain.invoke({"input": msg})
        print("Raw response:", response)

        raw_output = response.get("Full Response", response)
        cleaned_output = raw_output.split("Assistant:")[-1].strip()
        return cleaned_output
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "Something went wrong. Please try again."




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
