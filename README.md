Here is the generated `README.md` file based on the provided information:

---

# AI Sales Assistant

This repository contains a collection of Python scripts and Streamlit applications designed for creating an AI-powered sales assistant and interacting with data like PDFs and CSVs. Follow the instructions below to set up the project.

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/14b72deb-845d-4b58-859c-876744849738" />


---

## Prerequisites

1. **Install Required Packages**  
   Use `requirements.txt` to install all the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up API Keys**  
   To use the vector database, configure the following environment variables:
   - `PINECONE_ENVIRONMENT = us-east-1`
   - `PINECONE_INDEX_NAME = raghybridsearch`

   Add these variables to a `.env` file in the root directory.

---

## Features and Usage

### 1. **HTML Templates Initialization**
   Before running any application, ensure you initialize the HTML templates by running:
   ```bash
   python htmlTemplates.py
   ```
   This script is integrated with other components of the system.

### 2. **Sales Chatbot AI**
   The AI Sales Assistant allows you to interact through audio and text. To launch the chatbot:
   ```bash
   streamlit run sales_ai_assistant.py
   ```

### 3. **Interact with PDFs and CSVs**
   Use the `rag_hybridsearch.py` application to interact with PDF and CSV files. Launch the app with:
   ```bash
   streamlit run rag_hybridsearch.py
   ```

### 4. **No-Code Streamlit App Generator**
   Generate a no-code Streamlit application by running:
   ```bash
   streamlit run create_streamlit_app.py
   ```

---

## Project Structure

- `htmlTemplates.py`  
   Initializes and manages HTML templates required by other applications.

- `sales_ai_assistant.py`  
   A Streamlit application for interacting with the AI-powered Sales Assistant using text and audio inputs.

- `rag_hybridsearch.py`  
   A Streamlit application for querying and interacting with PDF and CSV files using a hybrid search approach.

- `create_streamlit_app.py`  
   A tool to generate a no-code Streamlit application.

- `.env`  
   Environment variables, including API keys and database configurations.

---

## Notes
- **Vector Database**: The project uses a vector database for hybrid search. The environment is set to `us-east-1` and the index name is `raghybridsearch`.
- Ensure all the required API keys are set up in the `.env` file before running the applications.
