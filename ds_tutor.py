import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.llms.base import LLM
from typing import Optional, List

# Custom LLM wrapper for Transformers
class TransformersLLM(LLM):
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "transformers"

# Set page configuration
st.set_page_config(page_title="AI Data Science Mentor", page_icon="🤓", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .sidebar-header {
        font-size: 1.5em;
        color: #ff7f0e;
    }
    .question-box {
        border: 2px solid #d3d3d3;
        border-radius: 10px;
        padding: 10px;
        background-color: #f9f9f9;
    }
    .answer-box {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 15px;
        background-color: #e6f3ff;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Initialize custom LLM and chain
llm = TransformersLLM(model=model, tokenizer=tokenizer)
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an expert data science mentor. Provide a clear, concise, and accurate answer to the following question. Use examples where helpful.

    Question: {question}

    Answer:
    """
)
chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to generate a response
def generate_response(question):
    try:
        response = chain.run(question)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again!"

# Main UI
def main():
    # Title
    st.markdown('<div class="main-title">AI Data Science Mentor</div>', unsafe_allow_html=True)
    st.write("Ask me anything about data science, and I’ll guide you with clear, practical answers!")

    # Sidebar with tips and examples
    st.sidebar.markdown('<div class="sidebar-header">Tips & Examples</div>', unsafe_allow_html=True)
    st.sidebar.write("""
    **Tips:**
    - Be specific for better answers!
    - Ask about concepts, code, or tools.

    **Example Questions:**
    - "Explain linear regression with an example."
    - "How do I handle missing data in Python?"
    - "What’s the difference between overfitting and underfitting?"
    """)
    st.sidebar.image("https://via.placeholder.com/150", caption="Data Science Rocks!")

    # Question input
    with st.form(key="question_form"):
        user_question = st.text_area(
            "Your Question:",
            height=120,
            placeholder="Type your data science question here...",
            key="question_input",
            help="Ask anything from basic stats to advanced ML!"
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.form_submit_button(label="Ask Now", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button(label="Clear", use_container_width=True)

    # Handle submission and clear
    if submit_button and user_question:
        with st.spinner("Mentoring in progress..."):
            response = generate_response(user_question)
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.subheader("Your Answer")
            st.write(response)
            st.markdown('</div>', unsafe_allow_html=True)
    elif clear_button:
        st.session_state.question_input = ""

if __name__ == "__main__":
    main()
