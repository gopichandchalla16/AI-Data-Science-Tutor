import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
import time

# Set Streamlit Page Config
st.set_page_config(page_title="AI Data Science Guru", page_icon="🚀", layout="wide")

# Load Gemini API Key
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("⚠️ Please set the GEMINI_API_KEY in Streamlit secrets to proceed!")
    st.stop()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

def speak(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

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

def listen():
    """Capture audio from the user and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "I couldn't understand. Please try again."
    except sr.RequestError:
        return "Could not request results. Please check your internet connection."

def analyze_file(uploaded_file):
    """Analyze uploaded CSV file and provide insights."""
    try:
        df = pd.read_csv(uploaded_file)
        summary = df.describe().to_string()
        return f"File Analysis:\n{summary}"
    except Exception as e:
        return f"⚠️ Error processing file: {str(e)}"

# Main Streamlit UI
def main():
    st.title("🚀 AI Conversational Data Science Tutor")
    st.markdown("### Your AI-powered tutor for Data Science & AI learning!")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])
    if uploaded_file:
        file_analysis = analyze_file(uploaded_file)
        st.text_area("📊 File Insights", value=file_analysis, height=200)

    # Speech or Text Input
    option = st.radio("Choose Input Method:", ["🎤 Speak", "⌨️ Type"], horizontal=True)

    user_question = ""
    if option == "🎤 Speak":
        if st.button("🎙️ Start Talking"):
            user_question = listen()
            st.write(f"You said: {user_question}")
    else:
        user_question = st.text_area("Ask Your Question:", placeholder="E.g. What is logistic regression?")

    # Generate Response
    if st.button("🤖 Get AI Response"):
        if user_question.strip():
            with st.spinner("AI is thinking..."):
                response = generate_response(user_question)
                st.success("AI Response:")
                st.write(response)
                speak(response)
        else:
            st.warning("⚠️ Please enter a question.")

    # Feedback Section
    st.subheader("📢 Give Your Feedback")
    feedback = st.text_area("How was your experience?")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! 🙌")

if __name__ == "__main__":
    main()
