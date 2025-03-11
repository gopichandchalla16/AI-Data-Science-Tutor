import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from gtts import gTTS
import os
import time

# Set Streamlit Page Config
st.set_page_config(page_title="AI Data Science Buddy", page_icon="📊", layout="wide")

# Custom CSS for Enhanced Styling
st.markdown("""
    <style>
    .main-title { font-size: 3em; color: #2ecc71; text-align: center; font-weight: bold; font-family: 'Arial', sans-serif; margin-bottom: 10px; }
    .subtitle { text-align: center; color: #7f8c8d; font-size: 1.2em; margin-bottom: 20px; }
    .chat-box { border: 2px solid #3498db; border-radius: 12px; padding: 20px; background: linear-gradient(135deg, #f5f7fa, #c3cfe2); box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .user-msg { background-color: #ecf0f1; padding: 10px; border-radius: 8px; margin: 5px 0; }
    .ai-msg { background-color: #2ecc71; color: white; padding: 10px; border-radius: 8px; margin: 5px 0; }
    .stButton>button { background-color: #e74c3c; color: white; font-size: 16px; padding: 12px 25px; border-radius: 8px; }
    .stButton>button:hover { background-color: #c0392b; }
    .stTextInput>div>input { border: 2px solid #3498db; border-radius: 8px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# Load Gemini API key
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("⚠️ Oops! Looks like the GEMINI_API_KEY is missing. Please set it in Streamlit secrets to get started!")
    st.stop()

# Custom LLM Wrapper for Gemini API
class GeminiLLM(LLM):
    def _call(self, prompt: str, stop=None):
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            return response.text if response and hasattr(response, "text") else "Hmm, I couldn’t fetch a response. Let’s try that again!"
        except Exception as e:
            return f"Uh-oh! Something went wrong: {str(e)}"

    def predict(self, prompt: str):
        return self._call(prompt)

    @property
    def _llm_type(self):
        return "Gemini"

# Initialize Gemini LLM
llm = GeminiLLM()

# Define the Prompt Template with a Talkative Personality
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    You are an enthusiastic and talkative AI Data Science Buddy! Your mission is to help users with their data science questions in a fun, friendly, and engaging way. 
    Be conversational, throw in some excitement, and make the user feel like they’re chatting with a knowledgeable friend. Keep it concise, accurate, and focused on data science—don’t wander off-topic!

    Conversation History:
    {history}

    User: {input}
    AI Buddy: Hey there! Alright, let’s dive into this! 
    """
)

# Initialize Conversation Memory
memory = ConversationBufferMemory()

# Create LangChain ConversationChain
conversation = ConversationChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory
)

# Function to Generate AI Response
def generate_response(user_input):
    try:
        response = conversation.run(input=user_input)
        return response
    except Exception as e:
        return f"Whoops! Something glitchy happened: {str(e)}"

# Function to Convert Text to Speech
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        # Cross-platform audio playback
        if os.name == 'nt':  # Windows
            os.system("start response.mp3")
        elif os.name == 'posix':  # macOS/Linux
            os.system("afplay response.mp3" if 'darwin' in os.uname().sysname.lower() else "mpg123 response.mp3")
    except Exception:
        st.warning("Oops! Text-to-speech didn’t work this time—let’s keep it text-only for now!")

# Main Streamlit UI
def main():
    st.markdown('<div class="main-title">AI Data Science Buddy</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your Chatty Data Science Pal—Here to Help with a Smile! 📈✨</div>', unsafe_allow_html=True)

    # Initialize session state for conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        ai_greeting = "Hey, hey! I’m your AI Data Science Buddy! Super excited to chat with you about all things data science. What’s on your mind today?"
        st.session_state.conversation_history.append({"user": "", "ai": ai_greeting})
        text_to_speech(ai_greeting)

    # Chat Interface
    st.markdown("### Let’s Chat!")
    chat_container = st.container()
    with chat_container:
        for exchange in st.session_state.conversation_history:
            if exchange["user"]:
                st.markdown(f'<div class="user-msg"><b>You:</b> {exchange["user"]}</div>', unsafe_allow_html=True)
            if exchange["ai"]:
                st.markdown(f'<div class="ai-msg"><b>AI Buddy:</b> {exchange["ai"]}</div>', unsafe_allow_html=True)

    # Input Section
    with st.form(key="input_form", clear_on_submit=True):
        user_input = st.text_input("Ask me anything about data science!", placeholder="E.g., What’s a decision tree?")
        submit_button = st.form_submit_button(label="Send It!")

    # Handle User Input
    if submit_button and user_input.strip():
        with st.spinner("Thinking up something awesome... 🤓"):
            ai_response = generate_response(user_input)
            st.session_state.conversation_history.append({"user": user_input, "ai": ai_response})
            chat_container.markdown(f'<div class="ai-msg"><b>AI Buddy:</b> {ai_response}</div>', unsafe_allow_html=True)
            text_to_speech(ai_response)
            time.sleep(0.5)  # Small delay to ensure UI updates smoothly
    elif submit_button and not user_input.strip():
        st.warning("Hey, don’t leave me hanging! Type something cool to ask!")

    # Fun Interactive Feature: Quick Tips Button
    if st.button("Gimme a Random Data Science Tip!"):
        tips = [
            "Did you know? Overfitting is like memorizing answers instead of learning the logic—keep your models simple!",
            "Pro tip: Always split your data into train and test sets—it’s like keeping a secret stash of candy to check later!",
            "Fun fact: A confusion matrix isn’t confusing once you see it as a scorecard for your model’s predictions!"
        ]
        random_tip = generate_response("Give me a quick random data science tip!")
        st.markdown(f'<div class="ai-msg"><b>AI Buddy:</b> {random_tip}</div>', unsafe_allow_html=True)
        text_to_speech(random_tip)

if __name__ == "__main__":
    main()
