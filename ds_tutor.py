import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

# Page Configuration
st.set_page_config(page_title="AI Data Science Guru", page_icon="🚀", layout="wide")

# Enhanced Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8em;
        color: #2ecc71;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
        animation: fadeIn 1.5s ease-in-out;
    }
    .chat-container {
        border-radius: 15px;
        padding: 20px;
        background: #f4f4f4;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 15px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Configure Gemini API
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("❌ Please set the GEMINI_API_KEY in Streamlit secrets to proceed!")
    st.stop()

# Define the Gemini Model
model = genai.GenerativeModel("gemini-1.5-flash")

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    Hey there! I’m your friendly AI Data Science Guru. You’ve asked: "{question}".
    Here’s a clear and engaging answer with examples and tips:
    """
)

# Custom Gemini Wrapper
class GeminiLLM:
    def __init__(self, model):
        self.model = model

    def __call__(self, prompt):  # Fixed Method
        response = self.model.generate_content(prompt)
        return response.text if hasattr(response, 'text') else "Sorry, I encountered an issue!"

# Initialize LLM Chain
llm = GeminiLLM(model)
chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to Generate Response
def generate_response(question):
    try:
        response = chain.run(question)
        response += "\n\n💡 Got more questions? I’m all ears!"
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}. Try again!"

# Main UI
st.markdown('<div class="main-title">AI Data Science Guru 🚀</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Your interactive, chatty AI tutor powered by Google Gemini!</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("💡 Guru’s Toolkit")
st.sidebar.markdown("""
- **Ask me anything:** Stats, ML, coding, you name it!
- **Want code?** Say "with code example"!
- **Try These:**
  - "Explain logistic regression with a fun example."
  - "How do I clean data in Python with code?"
  - "Why do neural networks work well?"
""")
if st.sidebar.button("🎲 Surprise Me!"):
    st.sidebar.success("Did you know? The term 'data science' was coined in 2001 by William S. Cleveland!")

# Chat Input
question = st.text_area("🤖 Ask Your Guru:", placeholder="What’s on your mind today?", height=100)

# Chat History State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Button
if st.button("💬 Chat Now!") and question:
    with st.spinner("Generating response..."):
        answer = generate_response(question)
        st.session_state.chat_history.append((question, answer))

# Display Chat History
if st.session_state.chat_history:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**🙋 You:** {q}")
        st.info(f"**🧠 Guru:** {a}")
    st.markdown("</div>", unsafe_allow_html=True)

# Clear Chat Button
if st.button("🔄 Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
