import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

# Set page configuration
st.set_page_config(page_title="AI Data Science Guru", page_icon="🚀", layout="wide")

# Enhanced Custom CSS with animations
st.markdown("""
    <style>
    .main-title {
        font-size: 3em;
        color: #2ecc71;
        text-align: center;
        font-family: 'Arial', sans-serif;
        animation: fadeIn 1.5s ease-in-out;
    }
    .sidebar-header {
        font-size: 1.8em;
        color: #e74c3c;
        font-weight: bold;
    }
    .question-box {
        border: 3px solid #3498db;
        border-radius: 15px;
        padding: 15px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .question-box:hover {
        transform: scale(1.02);
    }
    .answer-box {
        border: 3px dashed #2ecc71;
        border-radius: 15px;
        padding: 20px;
        background: linear-gradient(135deg, #dff9fb 0%, #b8e994 100%);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        margin-top: 25px;
        animation: slideIn 0.5s ease-in-out;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px 20px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    @keyframes slideIn {
        0% { transform: translateY(20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Configure Gemini API using Streamlit secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("Please set the GEMINI_API_KEY in Streamlit secrets to proceed!")
    st.stop()

# Define the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Enhanced prompt for a talkative, responsive tutor
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    Hey there! I’m your friendly AI Data Science Guru, here to chat about all things data! You’ve asked: "{question}". Let’s dive in with a clear, engaging, and detailed answer—think of me as your enthusiastic study buddy! I’ll throw in examples, tips, and maybe a fun fact or two. Ready? Here’s your answer:

    """
)

# Custom LLM wrapper for Gemini
class GeminiLLM:
    def __init__(self, model):
        self.model = model

    def _call(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text

# Initialize the chain
llm = GeminiLLM(model)
chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to generate a response with typing effect
def generate_response(question):
    try:
        response = chain.run(question)
        # Add a talkative outro
        response += "\n\nGot more questions? I’m all ears—let’s keep the data science party going!"
        return response
    except Exception as e:
        return f"Oops, I tripped over some code! Error: {str(e)}. Ask me again—I’ll get it right this time!"

# Main UI
def main():
    # Title with flair
    st.markdown('<div class="main-title">AI Data Science Guru</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Your chatty, data-loving tutor powered by Google Gemini!</p>", unsafe_allow_html=True)

    # Sidebar with enhanced interactivity
    st.sidebar.markdown('<div class="sidebar-header">Guru’s Toolkit</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    **Pro Tips:**
    - Ask me anything—stats, ML, coding, you name it!
    - Want code? Say "with code example"!
    - I love a good chat, so keep the questions coming!

    **Try These:**
    - "Teach me about logistic regression with a fun example."
    - "How do I clean data in Python with code example?"
    - "Why do neural networks work so well?"
    """)
    st.sidebar.image("https://via.placeholder.com/150?text=Data+Guru", caption="Your Data Buddy!")
    if st.sidebar.button("Surprise Me!"):
        st.sidebar.write("Here’s a fun fact: Did you know the term 'data science' was coined in 2001 by William S. Cleveland?")

    # Question input with enhanced styling
    with st.form(key="question_form"):
        user_question = st.text_area(
            "Ask Your Guru:",
            height=150,
            placeholder="What’s on your data-loving mind today?",
            key="question_input",
            help="I’m here to explain, chat, and code—go wild!"
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            submit_button = st.form_submit_button(label="Chat Now!")
        with col2:
            clear_button = st.form_submit_button(label="Start Fresh")

    # Handle submission with typing effect
    if submit_button and user_question:
        with st.spinner("Your Guru is typing..."):
            response = generate_response(user_question)
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.subheader("Guru Says:")
            # Simulate typing effect
            response_container = st.empty()
            for i in range(len(response)):
                response_container.write(response[:i+1])
                time.sleep(0.02)  # Adjust speed of typing effect
            st.markdown('</div>', unsafe_allow_html=True)
    elif clear_button:
        st.session_state.question_input = ""

if __name__ == "__main__":
    main()
