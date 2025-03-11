import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from gtts import gTTS
import io
import time
from datetime import datetime

# Streamlit Page Config
st.set_page_config(page_title="AI Data Science Tutor", page_icon="📊", layout="wide")

# Custom CSS (Inspired by React Components)
st.markdown("""
    <style>
    .main-title { font-size: 3em; color: #2ecc71; text-align: center; font-weight: bold; font-family: 'Arial', sans-serif; margin-bottom: 10px; }
    .subtitle { text-align: center; color: #7f8c8d; font-size: 1.2em; margin-bottom: 20px; }
    .card { border-radius: 12px; background: #000000e6; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); position: relative; overflow: hidden; }
    .spotlight { position: absolute; top: -40px; left: 0; width: 100%; height: 100%; background: radial-gradient(circle, rgba(255,255,255,0.2), transparent); z-index: 0; }
    .chat-box { border: 2px solid #3498db; border-radius: 12px; padding: 20px; background: linear-gradient(135deg, #f5f7fa, #c3cfe2); box-shadow: 0 4px 12px rgba(0,0,0,0.1); min-height: 400px; overflow-y: auto; margin-bottom: 20px; }
    .user-msg { background-color: #ecf0f1; padding: 12px; border-radius: 8px; margin: 8px 0; font-size: 1.1em; }
    .ai-msg { background-color: #27ae60; color: white; padding: 15px; border-radius: 10px; margin: 8px 0; font-size: 1.2em; font-weight: bold; animation: fadeIn 0.5s ease-in; }
    .did-you-know { position: fixed; right: 20px; top: 150px; width: 200px; background-color: #f1c40f; padding: 15px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-size: 0.9em; color: #2c3e50; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .stButton>button { background-color: #e74c3c; color: white; font-size: 16px; padding: 12px 25px; border-radius: 8px; margin: 5px; }
    .stButton>button:hover { background-color: #c0392b; }
    .stTextInput>div>input { border: 2px solid #3498db; border-radius: 8px; padding: 10px; font-size: 1.1em; }
    .footer { text-align: center; padding: 20px; background-color: #ecf0f1; border-top: 2px solid #3498db; font-size: 1em; color: #2c3e50; position: fixed; bottom: 0; width: 100%; box-shadow: 0 -2px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Load Gemini API Key
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("⚠️ Please set GEMINI_API_KEY in Streamlit secrets!")
    st.stop()

# Custom LLM Wrapper for Gemini
class GeminiLLM(LLM):
    def _call(self, prompt: str, stop=None):
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            return response.text if response and hasattr(response, "text") else "Hmm, I couldn’t fetch a response!"
        except Exception as e:
            return f"Error: {str(e)}"

    def predict(self, prompt: str):
        return self._call(prompt)

    @property
    def _llm_type(self):
        return "Gemini"

# Initialize LLM, Prompt, and Conversation
llm = GeminiLLM()
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    You are an enthusiastic AI Data Science Tutor! Help users with data science questions in a fun, friendly way. Keep answers concise, accurate, and engaging, with examples where possible. Stick to data science topics!

    History:
    {history}

    User: {input}
    AI Tutor: Hey there! Let’s dive in—here’s your answer:
    """
)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, prompt=prompt_template, memory=memory)

# Text-to-Speech Function
def text_to_speech(text):
    try:
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in ".,!?")
        if not clean_text.strip():
            clean_text = "Nothing to say—ask me something cool!"
        tts = gTTS(text=clean_text, lang='en', slow=False, tld='co.uk')  # British male-like voice
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except Exception as e:
        st.warning(f"Voice error: {str(e)}")
        return None

# Main App
def main():
    # Header (Inspired by SplineSceneBasic)
    st.markdown('<div class="main-title">AI Data Science Tutor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Your Interactive Data Science Buddy | {datetime.now().strftime("%B %d, %Y")}</div>', unsafe_allow_html=True)

    # Main Card Layout
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="spotlight"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    # Left Column: Intro and Chat
    with col1:
        st.markdown("""
            <div style="position: relative; z-index: 10; padding: 20px;">
                <h1 style="font-size: 2.5em; font-weight: bold; background-clip: text; -webkit-background-clip: text; color: transparent; background-image: linear-gradient(to bottom, #f7fafc, #a0aec0);">
                    Learn Data Science
                </h1>
                <p style="color: #d1d5db; max-width: 400px; margin-top: 16px;">
                    Ask me anything about data science—stats, coding, models—and I’ll guide you with enthusiasm!
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Chat Interface
        st.markdown("### Chat with Me!")
        with st.container():
            chat_container = st.empty()
            chat_html = ""
            if "history" not in st.session_state:
                st.session_state.history = [{"user": "", "ai": "Hey! I’m your Data Science Tutor—super excited to help! What’s on your mind?"}]
                st.session_state.last_audio = text_to_speech(st.session_state.history[0]["ai"])
            for exchange in st.session_state.history:
                if exchange["user"]:
                    chat_html += f'<div class="user-msg"><b>You:</b> {exchange["user"]}</div>'
                if exchange["ai"]:
                    chat_html += f'<div class="ai-msg"><b>Tutor:</b> {exchange["ai"]}</div>'
            chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
            chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)

    # Right Column: Input and Tools
    with col2:
        # Input Form
        with st.form(key="input_form", clear_on_submit=True):
            user_input = st.text_input("Ask a Question!", placeholder="E.g., What’s a decision tree?")
            submit_button = st.form_submit_button(label="Send It!")

        # Handle Input
        if submit_button and user_input.strip():
            with st.spinner("Thinking..."):
                response = conversation.run(input=user_input)
                st.session_state.history.append({"user": user_input, "ai": response})
                chat_html += f'<div class="user-msg"><b>You:</b> {user_input}</div>'
                chat_html += f'<div class="ai-msg"><b>Tutor:</b> {response}</div>'
                chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
                chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
                st.session_state.last_audio = text_to_speech(response)
                if st.session_state.last_audio:
                    st.audio(st.session_state.last_audio, format="audio/mp3")
        elif submit_button:
            st.warning("Give me something to work with!")

        # Interactive Features
        st.markdown("### Quick Tools")
        col3, col4, col5 = st.columns(3)
        with col3:
            if st.button("Random Tip", key="tip_btn"):
                tip = conversation.run("Give me a quick data science tip!")
                st.session_state.history.append({"user": "Random tip", "ai": tip})
                chat_html += f'<div class="user-msg"><b>You:</b> Random tip</div>'
                chat_html += f'<div class="ai-msg"><b>Tutor:</b> {tip}</div>'
                chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
                st.session_state.last_audio = text_to_speech(tip)
                if st.session_state.last_audio:
                    st.audio(st.session_state.last_audio, format="audio/mp3")
        with col4:
            if st.button("Repeat Last", key="repeat_btn"):
                if st.session_state.get("last_audio"):
                    st.audio(st.session_state.last_audio, format="audio/mp3")
                    st.success("Replaying last response!")
        with col5:
            if st.button("Clear Chat", key="clear_btn"):
                st.session_state.history = [{"user": "", "ai": "Chat cleared! Let’s start fresh!"}]
                st.session_state.last_audio = text_to_speech("Chat cleared! Let’s start fresh!")
                chat_container.markdown('<div class="chat-box"><div class="ai-msg"><b>Tutor:</b> Chat cleared! Let’s start fresh!</div></div>', unsafe_allow_html=True)

        # Did You Know Box
        st.markdown("""
            <div class="did-you-know">
                <b>Did You Know?</b><br>
                The term "Data Science" was coined in 2001 by William S. Cleveland!
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close card

    # Footer
    st.markdown("""
        <div class="footer">
            Built with ❤️ by Gopichand | Powered by Gemini AI
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
