import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from gtts import gTTS
import os
import time
from datetime import datetime

# Set Streamlit Page Config
st.set_page_config(page_title="AI Data Science Tutor", page_icon="📊", layout="wide")

# Custom CSS inspired by React component and your styling
st.markdown("""
    <style>
    .main-title { font-size: 3em; color: #2ecc71; text-align: center; font-weight: bold; font-family: 'Arial', sans-serif; margin-bottom: 10px; }
    .subtitle { text-align: center; color: #7f8c8d; font-size: 1.2em; margin-bottom: 20px; }
    .chat-box { border: 2px solid #3498db; border-radius: 12px; padding: 20px; background: linear-gradient(135deg, #f5f7fa, #c3cfe2); box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px; min-height: 400px; overflow-y: auto; }
    .user-msg { background-color: #ecf0f1; padding: 12px; border-radius: 8px; margin: 8px 0; font-size: 1.1em; }
    .ai-msg { background-color: #27ae60; color: white; padding: 15px; border-radius: 10px; margin: 8px 0; font-size: 1.2em; font-weight: bold; animation: fadeIn 0.5s ease-in; }
    .spotlight { position: absolute; top: -40px; left: 0; width: 100%; height: 100%; background: radial-gradient(circle, rgba(255,255,255,0.2), transparent); }
    .interactive-btn { background-color: #3498db; color: white; padding: 10px 20px; border-radius: 8px; }
    .interactive-btn:hover { background-color: #2980b9; }
    .footer { text-align: center; padding: 15px; background-color: #ecf0f1; border-top: 2px solid #3498db; font-size: 1em; color: #2c3e50; position: fixed; bottom: 0; width: 100%; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
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
            return response.text if response and hasattr(response, "text") else "Sorry, I couldn’t fetch a response!"
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
    You are an AI Data Science Tutor with a friendly, enthusiastic vibe! Help users with data science questions, explain concepts clearly, and keep it fun. Stick to data science topics and use examples where possible.

    History:
    {history}

    User: {input}
    AI Tutor: Hey there! Let’s tackle this together—here’s what I’ve got for you:
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
    # Header inspired by SplineSceneBasic
    st.markdown('<div class="main-title">AI Data Science Tutor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Your Interactive Guide to Data Science Mastery | {datetime.now().strftime("%B %d, %Y")}</div>', unsafe_allow_html=True)

    # Chat Container
    st.markdown("### Chat with Your Tutor")
    with st.container():
        chat_container = st.empty()
        chat_html = ""
        if "history" not in st.session_state:
            st.session_state.history = [{"user": "", "ai": "Hi! I’m your Data Science Tutor—ready to dive into stats, code, or whatever’s on your mind? What’s up?"}]
            st.session_state.last_audio = text_to_speech(st.session_state.history[0]["ai"])
        for exchange in st.session_state.history:
            if exchange["user"]:
                chat_html += f'<div class="user-msg"><b>You:</b> {exchange["user"]}</div>'
            if exchange["ai"]:
                chat_html += f'<div class="ai-msg"><b>Tutor:</b> {exchange["ai"]}</div>'
        chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
        chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)

    # Input Section
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.form(key="input_form", clear_on_submit=True):
            user_input = st.text_input("Ask me anything about data science!", placeholder="E.g., Explain linear regression")
            submit_button = st.form_submit_button(label="Ask!")
    with col2:
        if st.session_state.get("last_audio"):
            st.audio(st.session_state.last_audio, format="audio/mp3")

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

    # Interactive Features (inspired by React demo’s interactivity)
    st.markdown("### Quick Tools")
    col3, col4, col5 = st.columns(3)
    with col3:
        if st.button("Random Data Science Tip", key="tip_btn"):
            tip = conversation.run("Give me a quick data science tip!")
            st.session_state.history.append({"user": "Random tip", "ai": tip})
            chat_html += f'<div class="user-msg"><b>You:</b> Random tip</div>'
            chat_html += f'<div class="ai-msg"><b>Tutor:</b> {tip}</div>'
            chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
            st.session_state.last_audio = text_to_speech(tip)
    with col4:
        if st.button("Repeat Last", key="repeat_btn"):
            if st.session_state.get("last_audio"):
                st.audio(st.session_state.last_audio, format="audio/mp3")
    with col5:
        if st.button("Clear Chat", key="clear_btn"):
            st.session_state.history = [{"user": "", "ai": "Chat cleared! What’s next?"}]
            st.session_state.last_audio = text_to_speech("Chat cleared! What’s next?")
            chat_container.markdown('<div class="chat-box"><div class="ai-msg"><b>Tutor:</b> Chat cleared! What’s next?</div></div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">Built with ❤️ by Gopichand | Powered by Gemini AI</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
