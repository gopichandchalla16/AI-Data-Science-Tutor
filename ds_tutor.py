import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
import speech_recognition as sr
from gtts import gTTS
import os
import time
import pandas as pd
import base64

# Streamlit Page Configuration
st.set_page_config(page_title="AI Data Science Tutor", page_icon="🎓", layout="wide")

# Load Gemini API Key
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("⚠️ Please set the GEMINI_API_KEY in Streamlit secrets!")
    st.stop()

# Custom LLM Wrapper for Gemini API
class GeminiLLM(LLM):
    def _call(self, prompt: str, stop=None):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text if response and hasattr(response, "text") else "⚠️ No response from AI."
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

    def predict(self, prompt: str):
        return self._call(prompt)

    @property
    def _llm_type(self):
        return "Gemini"

# Initialize Gemini LLM
llm = GeminiLLM()

# Define Prompt Template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="You're an expert AI tutor. Answer the following:\n\nQuestion: {question}\n\nAnswer:"
)

# Create LangChain Chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to Generate AI Response
def generate_response(question):
    try:
        response = chain.run(question)
        return response
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# Function for AI Voice Response
def speak_response(response_text):
    tts = gTTS(response_text)
    tts.save("response.mp3")
    return "response.mp3"

# Function for Speech Recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"🗣 You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("⚠️ Could not understand the audio.")
        except sr.RequestError:
            st.error("⚠️ Speech recognition service unavailable.")
    return ""

# Function to Process Uploaded CSV File
def process_csv(file):
    df = pd.read_csv(file)
    st.write("📊 Uploaded Data Preview:")
    st.dataframe(df.head())

    # Provide a summary
    st.write("📈 **Data Summary:**")
    st.write(df.describe())

    return df

# Main Streamlit UI
def main():
    st.title("📚 AI Data Science Tutor - Conversational Mode")
    st.write("💡 Talk to your AI tutor or upload a dataset for insights.")

    # Speech Input Button
    if st.button("🎤 Speak to AI"):
        user_question = recognize_speech()
        if user_question:
            with st.spinner("💭 AI is thinking..."):
                response = generate_response(user_question)
                st.success(f"🤖 AI: {response}")
                audio_file = speak_response(response)
                st.audio(audio_file, format="audio/mp3")

    # Text Input Box
    user_question = st.text_area("📌 Ask your AI Tutor:", placeholder="E.g. Explain Gradient Boosting...")

    # Chat Button
    if st.button("💬 Chat Now!"):
        if user_question.strip():
            with st.spinner("💭 AI is thinking..."):
                response = generate_response(user_question)
                st.markdown(f"**🤖 AI Response:**\n\n{response}")
                audio_file = speak_response(response)
                st.audio(audio_file, format="audio/mp3")
        else:
            st.warning("⚠️ Please enter a question.")

    # File Upload Section
    uploaded_file = st.file_uploader("📂 Upload a CSV file for analysis", type=["csv"])
    if uploaded_file is not None:
        process_csv(uploaded_file)

    # Feedback Section
    st.subheader("💬 Feedback")
    feedback = st.text_area("How can we improve this AI Tutor?")
    if st.button("📩 Submit Feedback"):
        st.success("✅ Thank you for your feedback!")

if __name__ == "__main__":
    main()
