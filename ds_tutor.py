import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set page configuration
st.set_page_config(page_title="AI Data Science Tutor", page_icon="📊", layout="wide")

# Title and description
st.title("AI Data Science Tutor")
st.markdown("""
Welcome to your personal Data Science Tutor! Ask me anything about data science, machine learning, statistics, or coding, and I'll do my best to help you learn.
""")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "distilgpt2"  # Lightweight model for demo purposes
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Define the prompt template for LangChain
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an expert data science tutor. Provide a clear, concise, and accurate answer to the following question. Use examples where helpful.

    Question: {question}

    Answer:
    """
)

# Function to generate a response
def generate_response(question):
    # Tokenize input
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate output with the model
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode the response
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Use LangChain to structure the response
    chain = LLMChain(llm=None, prompt=prompt_template)  # We'll manually inject the response
    formatted_response = prompt_template.format(question=question) + raw_response
    return formatted_response

# Streamlit UI
def main():
    # Input form
    with st.form(key="question_form"):
        user_question = st.text_area("Ask your data science question here:", height=100)
        submit_button = st.form_submit_button(label="Get Answer")

    # Process the question and display the answer
    if submit_button and user_question:
        with st.spinner("Thinking..."):
            response = generate_response(user_question)
            st.subheader("Answer")
            st.write(response)
    
    # Sidebar with examples
    st.sidebar.header("Example Questions")
    st.sidebar.write("""
    - "What is the difference between supervised and unsupervised learning?"
    - "How does a decision tree work?"
    - "Explain the bias-variance tradeoff with an example."
    """)

if __name__ == "__main__":
    main()
