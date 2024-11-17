import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone, ServerlessSpec
import streamlit as st

# Load environment variables
load_dotenv()

# Pinecone API setup
api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not api_key or not environment or not index_name:
    raise ValueError("Ensure PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME are properly set in the .env file.")

pinecone_client = Pinecone(api_key=api_key)
available_indexes = pinecone_client.list_indexes()

if index_name not in [idx.name for idx in available_indexes]:
    pinecone_client.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=environment
        )
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' exists. Connecting to the index.")

# Functions
def get_pdf_text(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF.")
    return text

def get_csv_text(csv_file):
    """Extract text from a CSV file."""
    df = pd.read_csv(csv_file)
    text = df.to_string(index=False)  # Convert the CSV data into a readable string
    return text

def split_text_into_chunks(text):
    """Split extracted text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def store_in_pinecone(chunks):
    """Generate embeddings and store them in Pinecone using Langchain."""
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = LangchainPinecone.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=""  # Use a specific namespace if needed
    )
    print("Embeddings stored in Pinecone.")
    return vectorstore

def get_conversation_chain(vectorstore):
    """Set up a conversational retrieval chain."""
    llm = ChatOpenAI(temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

# Streamlit App
def main():
    st.set_page_config(page_title="File Chat with Pinecone", page_icon="📄")
    st.title("Chat with your File 📄")
    st.sidebar.header("Upload File")
    
    uploaded_file = st.sidebar.file_uploader("Upload your file (PDF or CSV)", type=["pdf", "csv"])
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                st.info("Extracting text from PDF...")
                text = get_pdf_text(uploaded_file)
                st.success("Text successfully extracted!")
            elif uploaded_file.type == "text/csv":
                st.info("Extracting text from CSV...")
                text = get_csv_text(uploaded_file)
                st.success("Text successfully extracted!")
            else:
                st.error("Unsupported file type.")
                return
            
            st.info("Splitting text into chunks...")
            chunks = split_text_into_chunks(text)
            st.success(f"Generated {len(chunks)} chunks.")
            
            st.info("Storing embeddings in Pinecone...")
            vectorstore = store_in_pinecone(chunks)
            st.success("Embeddings successfully stored in Pinecone!")
            
            st.info("Setting up conversation...")
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Conversation chain ready! Ask your questions below.")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    if st.session_state.conversation:
        user_question = st.text_input("Ask a question:")
        if user_question:
            try:
                st.info("Fetching results from Pinecone...")
                response = st.session_state.conversation({"question": user_question})
                answer = response.get("answer", "No answer available.")
                st.session_state.chat_history.append({"user": user_question, "assistant": answer})
                st.write(f"**Assistant:** {answer}")
            except Exception as e:
                st.error(f"An error occurred while processing your question: {e}")
        
        if st.session_state.chat_history:
            st.write("### Chat History")
            for chat in st.session_state.chat_history:
                st.write(f"**You:** {chat['user']}")
                st.write(f"**Assistant:** {chat['assistant']}")

if __name__ == "__main__":
    main()