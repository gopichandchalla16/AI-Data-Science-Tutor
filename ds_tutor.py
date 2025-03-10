import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
import time
import base64
import speech_recognition as sr
import pyttsx3

# Set Streamlit Page Config
st.set_page_config(page_title="AI Data Science Guru", page_icon="🚀", layout="wide")

# Custom CSS for Improved UI
st.markdown("""
    <style>
    body {background-color: #f4f6f7; font-family: 'Arial', sans-serif;}
    .main-title { font-size: 3em; color: #1abc9c; text-align: center; font-weight: bold; }
    .sub-title { text-align: center; color: #7f8c8d; font-size: 1.2em; }
    .answer-box { border: 2px solid #1abc9c; border-radius: 12px; padding: 15px; background: #ecf0f1; box-shadow: 0 4px 8px rgba(0,0,0,0.15); margin-top: 20px; }
    .stButton>button { background-color: #3498db; color: white; font-size: 16px; padding: 10px 20px; border-radius: 8px; }
    .stButton>button:hover { background-color: #2980b9; }
    .mic-button { background-color: #e74c3c; color: white; padding: 10px 15px; border-radius: 50%; font-size: 20px; border: none; cursor: pointer; margin-right: 10px; }
    .mic-button:hover { background-color: #c0392b; }
    </style>
""", unsafe_allow_html=True)

# Load Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Custom LLM Wrapper for Gemini API
class GeminiLLM(LLM):
    def _call(self, prompt: str, stop=None):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text if response and hasattr(response, "text") else "⚠️ No response from Gemini API."
        except Exception as e:
            return f"⚠️ Error generating response: {str(e)}"
    
    def predict(self, prompt: str):
        return self._call(prompt)
    
    @property
    def _llm_type(self):
        return "Gemini"

# Initialize Gemini LLM
llm = GeminiLLM()

# Define the Prompt Template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="You're an expert AI tutor. Answer the following question in detail:\n\nQuestion: {question}\n\nAnswer:"
)

# Create LangChain LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)

def generate_response(question):
    try:
        response = chain.run(question)
        return response
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# Speech-to-Text Function (Mic Input)
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak your question...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "⚠️ Could not understand the audio. Try again."
    except sr.RequestError:
        return "⚠️ Could not request results. Check internet connection."

# Text-to-Speech Function (AI Voice Response)
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()

# Main Streamlit UI
def main():
    st.markdown('<div class="main-title">AI Data Science Guru</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Your AI-powered Data Science tutor! 🚀</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🎤 Speak Now!"):
            question_from_mic = recognize_speech()
            st.text_input("You said:", value=question_from_mic, key="mic_input", disabled=True)
    
    with col2:
        user_question = st.text_area("Ask Your Guru:", placeholder="E.g. What is logistic regression?", height=150)
    
    if st.button("Chat Now!"):
        if user_question.strip():
            with st.spinner("Thinking... 💭"):
                response = generate_response(user_question)
                st.markdown(f'<div class="answer-box"><b>Guru Says:</b><br>{response}</div>', unsafe_allow_html=True)
                
                # AI Speaks the Response
                st.button("🔊 Listen", on_click=speak_text, args=(response,))
        else:
            st.warning("⚠️ Please enter a question.")

if __name__ == "__main__":
    main()
