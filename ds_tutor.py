import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from datetime import datetime

# --- Custom CSS for Aesthetic UI ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #e0f7fa, #b2ebf2);
        font-family: 'Segoe UI', sans-serif;
        padding: 20px;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #ffffff;
        border-left: 6px solid #0288d1;
        max-width: 70%;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #e8f5e9;
        border-left: 6px solid #4caf50;
        max-width: 70%;
        margin-right: auto;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #0288d1;
        color: white;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff5722;
        transform: scale(1.05);
    }
    .stTextInput>div>input {
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #0288d1;
    }
    .header {
        text-align: center;
        color: #0277bd;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .subheader {
        text-align: center;
        color: #555;
        font-size: 1.2em;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #777;
        font-size: 0.9em;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Streamlit UI Setup ---
st.markdown("<div class='header'>📊 AI Data Science Tutor</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Your chatty, 100% free companion for data science!</div>", unsafe_allow_html=True)
st.write(f"📅 Today is {datetime.now().strftime('%B %d, %Y')}")
st.info("💡 Powered by DistilGPT2—completely free and local, no API costs!")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey there! I’m your Data Science Tutor, powered by DistilGPT2—100% free and running right on your machine! I’m pumped to chat about data, code, and all things geeky. What’s sparking your curiosity today? Let’s dive into the data science world together!"}
    ]
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# --- Load the DistilGPT2 Model ---
@st.cache_resource
def load_model():
    try:
        # Load DistilGPT2 locally (no API key needed)
        model = pipeline("text-generation", model="distilgpt2", max_length=200, truncation=True)
        llm = HuggingFacePipeline(pipeline=model)
        st.sidebar.success("DistilGPT2 Loaded 🎉 (100% Free)")
        return llm
    except Exception as e:
        st.sidebar.error(f"Failed to load DistilGPT2: {str(e)}")
        return None

llm = load_model()
if not llm:
    st.error("Can’t load DistilGPT2. Check your setup!")
    st.stop()

# --- Initialize LangChain Conversation Chain ---
conversation = ConversationChain(llm=llm, memory=st.session_state.memory, verbose=False)

# --- Sidebar Options ---
st.sidebar.title("✨ Tutor Settings")
st.sidebar.markdown("Customize your experience!")
explain_option = st.sidebar.checkbox("Detailed Explanations", value=True, help="Get extra details!")
code_option = st.sidebar.checkbox("Code Examples", value=True, help="See Python snippets!")
st.sidebar.markdown("---")
st.sidebar.write("💡 Running locally with DistilGPT2—no limits, no costs!")

# --- Display Chat History ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤓"):
            if "```" in message["content"]:
                parts = message["content"].split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        st.markdown(part)
                    else:
                        st.code(part, language="python")
            else:
                st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask me a data science question!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

    data_science_keywords = [
        "data", "science", "machine learning", "statistics", "python", "pandas", "numpy", 
        "regression", "model", "algorithm", "visualization", "sklearn", "tensorflow", 
        "deep learning", "probability", "hypothesis", "clustering", "sql", "matplotlib"
    ]
    is_data_science = any(keyword in prompt.lower() for keyword in data_science_keywords)

    with chat_container:
        with st.chat_message("assistant", avatar="🤓"):
            if is_data_science:
                base_response = conversation.predict(input=prompt)
                response = f"Wow, awesome question! {base_response} I’m totally geeking out over data science—it’s so much fun to explore! What do you think about this? Got any projects where this could shine?"

                if explain_option:
                    response += "\n\n**Let’s Dig In**: I’ll break it down like we’re chatting over a coffee. Ready for the full scoop?"
                if code_option and ("how" in prompt.lower() or "example" in prompt.lower()):
                    if "pandas" in prompt.lower():
                        response += "\n\nHere’s a cool Pandas example:\n```python\nimport pandas as pd\ndf = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Score': [85, 92]})\nprint(df[df['Score'] > 90])\n```\nFilters like magic, right?"
                    elif "regression" in prompt.lower():
                        response += "\n\nCheck out this regression snippet:\n```python\nfrom sklearn.linear_model import LinearRegression\nX = [[1], [2], [3]]\ny = [2, 4, 6]\nmodel = LinearRegression().fit(X, y)\nprint(model.predict([[4]]))\n```\nPredicting stuff is the best!"

                st.markdown(response)
            else:
                response = "Hmm, that’s not quite a data science vibe! I’m here to nerd out about data, stats, and coding. What data science topic can I jump into for you?"
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- Clear Chat Button ---
if st.button("🧹 Reset Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey there! I’m your Data Science Tutor, powered by DistilGPT2—100% free and running right on your machine! I’m pumped to chat about data, code, and all things geeky. What’s sparking your curiosity today? Let’s dive into the data science world together!"}
    ]
    st.session_state.memory.clear()
    st.experimental_rerun()

# --- Footer ---
st.markdown("<div class='footer'>💻 Powered by DistilGPT2, LangChain, and Streamlit | 100% Free & Local</div>", unsafe_allow_html=True)
