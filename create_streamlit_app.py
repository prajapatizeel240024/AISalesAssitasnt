import streamlit as st
import openai
import os
from dotenv import load_dotenv
import requests
import json
import subprocess
import time

load_dotenv()
error_code='''
def resolve_error_with_chatgpt(error_message, user_requirements, previous_code=""):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }
    url = "https://api.openai.com/v1/chat/completions"
    messages = [
        {"role": "system", "content": "You are an expert Python developer specializing in Streamlit applications."},
        {"role": "user", "content": f"Fix the following error for this Streamlit app requirement: '{user_requirements}'. Error details: {error_message}. Previous code: {previous_code}. Provide only the Streamlit code with no comments and any other text."}
    ]
    data = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.1,
        "top_p": 1.0
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    response_data = response.json()
    generated_code = response_data['choices'][0]['message']['content']
    generated_code = generated_code.strip("python```").strip("```")
    return generated_code

def generate_streamlit_code(user_requirements):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }
    url = "https://api.openai.com/v1/chat/completions"
    messages = [
        {"role": "system", "content": "You are an expert Python developer specializing in Streamlit applications."},
        {"role": "user", "content": f"Build a Streamlit app based on these requirements: {user_requirements} and make it robust like. Provide only the Streamlit code with no comments and any other text."}
    ]
    data = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.1,
        "top_p": 1.0
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    response_data = response.json()
    generated_code = response_data['choices'][0]['message']['content']
    generated_code = generated_code.strip("python```").strip("```")
    return generated_code

def install_dependencies(code):
    imports = [
        line.split()[1]
        for line in code.split("\n")
        if line.startswith("import ") or line.startswith("from ")
    ]
    unique_imports = list(set(imports))
    for package in unique_imports:
        try:
            subprocess.check_call(["pip", "install", package])
        except Exception as e:
            st.error(f"Error installing package {package}: {e}")

def run_app_and_resolve_errors(user_requirements):
    retries = 5  # Maximum retries to resolve errors
    app_code = generate_streamlit_code(user_requirements)
    for attempt in range(retries):
        try:
            with open("user_generated_app.py", "w") as f:
                f.write(app_code)

            st.info("Installing dependencies...")
            install_dependencies(app_code)

            st.info("Running the app...")
            os.system("streamlit run user_generated_app.py &")
            break
        except Exception as error:
            st.error(f"Error occurred: {error}. Resolving with ChatGPT...")
            app_code = resolve_error_with_chatgpt(
                str(error), user_requirements, previous_code=app_code
            )
    else:
        st.error("Maximum retries reached. Could not resolve the issue.")

    return app_code

st.title("No-Code Streamlit App Builder")
st.subheader("Describe what you want your app to do:")
user_input = st.text_area(
    "Enter your app requirements (e.g., 'A dashboard for visualizing CSV data with filtering options')",
    height=150
)

if "app_code" not in st.session_state:
    st.session_state.app_code = ""

if st.button("Generate App"):
    if user_input:
        with st.spinner("Generating your app..."):
            app_code = run_app_and_resolve_errors(user_input)
            st.session_state.app_code = app_code
            st.success("Your app was generated successfully!")

st.subheader("Generated Code:")
if st.session_state.app_code:
    st.code(st.session_state.app_cod
'''
def resolve_error_with_chatgpt(error_message, user_requirements, previous_code=""):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }
    url = "https://api.openai.com/v1/chat/completions"
    messages = [
        {"role": "system", "content": "You are an expert Python developer specializing in Streamlit applications."},
        {"role": "user", "content": f"Fix the following error for this Streamlit app requirement: '{user_requirements}'. Error details: {error_message}. Previous code: {previous_code}. Provide only the Streamlit code with no comments and any other text."}
    ]
    data = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.1,
        "top_p": 1.0
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    response_data = response.json()
    generated_code = response_data['choices'][0]['message']['content']
    generated_code = generated_code.strip("python```").strip("```")
    return generated_code

def generate_streamlit_code(user_requirements):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }
    url = "https://api.openai.com/v1/chat/completions"
    messages = [
        {"role": "system", "content": "You are an expert Python developer specializing in Streamlit applications."},
        {"role": "user", "content": f"Build a Streamlit app based on these requirements: {user_requirements}. The app should only have similar feedback framework to chatgpt to resolve any runtime errors and re-run the app. Provide only the Streamlit code with no comments and any other text."}
    ]
    data = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.1,
        "top_p": 1.0
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
    response_data = response.json()
    generated_code = response_data['choices'][0]['message']['content']
    generated_code = generated_code.strip("python```").strip("```")
    return generated_code

def install_dependencies(code):
    imports = [
        line.split()[1]
        for line in code.split("\n")
        if line.startswith("import ") or line.startswith("from ")
    ]
    unique_imports = list(set(imports))
    for package in unique_imports:
        try:
            subprocess.check_call(["pip", "install", package])
        except Exception as e:
            st.error(f"Error installing package {package}: {e}")

def run_app_and_resolve_errors(user_requirements):
    retries = 5  # Maximum retries to resolve errors
    app_code = generate_streamlit_code(user_requirements)
    for attempt in range(retries):
        try:
            with open("user_generated_app.py", "w") as f:
                f.write(app_code)

            st.info("Installing dependencies...")
            install_dependencies(app_code)

            st.info("Running the app...")
            os.system("streamlit run user_generated_app.py &")
            break
        except Exception as error:
            st.error(f"Error occurred: {error}. Resolving with ChatGPT...")
            app_code = resolve_error_with_chatgpt(
                str(error), user_requirements, previous_code=app_code
            )
    else:
        st.error("Maximum retries reached. Could not resolve the issue.")

    return app_code

st.title("No-Code Streamlit App Builder")
st.subheader("Describe what you want your app to do:")
user_input = st.text_area(
    "Enter your app requirements (e.g., 'A dashboard for visualizing CSV data with filtering options')",
    height=150
)

if "app_code" not in st.session_state:
    st.session_state.app_code = ""

if st.button("Generate App"):
    if user_input:
        with st.spinner("Generating your app..."):
            app_code = run_app_and_resolve_errors(user_input)
            st.session_state.app_code = app_code
            st.success("Your app was generated successfully!")

st.subheader("Generated Code:")
if st.session_state.app_code:
    st.code(st.session_state.app_code, language="python")
