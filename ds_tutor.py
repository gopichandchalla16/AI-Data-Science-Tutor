import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
import time

# Set Streamlit Page Config
st.set_page_config(page_title="AI Data Science Guru", page_icon="🚀", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.8em; color: #2ecc71; text-align: center; font-weight: bold; font-family: 'Arial', sans-serif; }
    .answer-box { border: 2px solid #2ecc71; border-radius: 12px; padding: 15px; background: linear-gradient(135deg, #e3f2fd, #b8e994); box-shadow: 0 4px 8px rgba(0,0,0,0.15); margin-top: 20px; }
    .stButton>button { background-color: #3498db; color: white; font-size: 16px; padding: 10px 20px; }
    .stButton>button:hover { background-color: #2980b9; }
    </style>
""", unsafe_allow_html=True)

# Load Gemini API key
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("⚠️ Please set the GEMINI_API_KEY in Streamlit secrets to proceed!")
    st.stop()

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

# Function to Generate AI Response
def generate_response(question):
    try:
        response = chain.run(question)
        return response
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# Main Streamlit UI
def main():
    st.markdown('<div class="main-title">AI Data Science Guru</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Your AI-powered Data Science tutor! 🚀</p>", unsafe_allow_html=True)

    # Input Box
    user_question = st.text_area("Ask Your Guru:", placeholder="E.g. What is logistic regression?", height=150)

    # Button
    if st.button("Chat Now!"):
        if user_question.strip():
            with st.spinner("Thinking... 💭"):
                response = generate_response(user_question)
                st.markdown(f'<div class="answer-box"><b>Guru Says:</b><br>{response}</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a question.")

if __name__ == "__main__":
    main()
