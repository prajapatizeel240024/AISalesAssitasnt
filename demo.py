import torch
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS
import pygame
from langdetect import detect
from deep_translator import GoogleTranslator
import os
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
import ssl
import time

# Load environment variables
load_dotenv()

# Initialize Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"The device used is {device}")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")


def record_audio(duration=10, sample_rate=16000):
    """Records audio for a given duration and sample rate"""
    print(f"Recording for {duration} seconds... Please speak clearly.")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio_data = np.squeeze(audio_data)
    print("Audio recording complete.")
    return audio_data, sample_rate


def transcribe_audio(audio_data, sample_rate):
    """Transcribe audio using Hugging Face Whisper"""
    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    detected_lang = detect(transcription)
    print(f"Transcription: {transcription} (Language: {detected_lang})")
    return transcription, detected_lang


def text_to_speech(text, lang="en"):
    """Convert text to speech using gTTS"""
    file_path = os.path.join(os.getcwd(), f"{lang}_output.mp3")
    tts = gTTS(text=text, lang=lang)
    tts.save(file_path)

    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    os.remove(file_path)


def translate_text(text, target_lang="en"):
    """Translate text using Deep Translator"""
    translator = GoogleTranslator(source="auto", target=target_lang)
    translation = translator.translate(text)
    print(f"Translated Text: {translation}")
    return translation


def lookup(question):
    """AI Agent Logic for Sales Assistant Without Search API"""
    llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    summary_template = """"
    You are a highly detailed Sales Assistant AI. Your task is to provide actionable insights about the target company, 
    domain, competitors, or market trends. Ensure your response is comprehensive, well-structured, and provides 
    relevant information for up to 500-600 words.

    Question: {question}

    Begin your response:
    """
    summary_prompt_template = PromptTemplate(input_variables=["question"], template=summary_template)
    response = llm(summary_prompt_template.format(question=question))
    print("LLM Output:", response)
    return response


def ai_agent(question):
    """AI Agent to process sales-related questions"""
    ssl._create_default_https_context = ssl._create_unverified_context

    # Perform the lookup using OpenAI's LLM
    response = lookup(question=question)

    # Extract the content of the AI response
    if hasattr(response, "content"):
        return response.content
    else:
        raise ValueError("Unexpected response format from AI agent.")


def main():
    """Main function to handle user input and integrate all components"""
    print("Welcome to Sales AI Assistant!")
    print("Options:")
    print("1. Speak your query")
    print("2. Enter your query as text")

    choice = input("Choose an option (1/2): ")

    if choice == "1":
        # Record and transcribe audio
        audio_data, sample_rate = record_audio(duration=10, sample_rate=16000)
        transcription, detected_lang = transcribe_audio(audio_data, sample_rate)

        # Translate if necessary
        if detected_lang != "en":
            question = translate_text(transcription, target_lang="en")
        else:
            question = transcription

        print(f"Processing your query: {question}")
        ai_response = ai_agent(question)

        # Text-to-speech for AI response
        print("Playing AI response...")
        text_to_speech(ai_response, lang="en")

    elif choice == "2":
        # Text-based query
        question = input("Enter your sales-related query: ")
        ai_response = ai_agent(question)

        # Check if the AI response is valid
        if not isinstance(ai_response, str):
            raise ValueError("AI response is not a valid string.")

        # Text-to-speech for AI response
        print("Playing AI response...")
        text_to_speech(ai_response, lang="en")

    else:
        print("Invalid choice! Exiting.")


if __name__ == "__main__":
    main()