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
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import os

# Streamlit Page Config
st.set_page_config(page_title="AI Data Science Robot", page_icon="🤖", layout="wide")

# Custom CSS (unchanged for brevity, but optimized for load time)
st.markdown("""
    <style>
    /* Same as your original CSS, minified for efficiency */
    .main-title{font-size:3.5em;color:#00d4ff;text-align:center;font-weight:bold;font-family:'Orbitron',sans-serif;text-shadow:0 0 10px #00d4ff,0 0 20px #00d4ff;margin-bottom:10px;}
    .subtitle{text-align:center;color:#b0bec5;font-size:1.3em;font-family:'Roboto Mono',monospace;text-shadow:0 0 5px #b0bec5;margin-bottom:20px;}
    .card{border-radius:15px;background:linear-gradient(145deg,#1a1a2e,#16213e);padding:25px;box-shadow:0 8px 16px rgba(0,0,0,0.5),inset 0 0 10px rgba(0,212,255,0.3);position:relative;overflow:hidden;transform:perspective(1000px) rotateX(2deg) rotateY(2deg);transition:transform 0.3s ease;}
    .card:hover{transform:perspective(1000px) rotateX(0deg) rotateY(0deg) scale(1.02);}
    .chat-box{border:3px solid #00d4ff;border-radius:12px;padding:20px;background:linear-gradient(135deg,#0f172a,#1e293b);box-shadow:0 4px 12px rgba(0,0,0,0.5),inset 0 0 8px rgba(0,212,255,0.2);min-height:400px;overflow-y:auto;}
    .user-msg{background:#334155;padding:12px;border-radius:8px;margin:8px 0;font-size:1.1em;color:#e2e8f0;box-shadow:0 2px 4px rgba(0,0,0,0.3);}
    .ai-msg{background:linear-gradient(90deg,#00d4ff,#007acc);color:#fff;padding:15px;border-radius:10px;margin:8px 0;font-size:1.2em;font-weight:bold;box-shadow:0 4px 8px rgba(0,0,0,0.4);animation:fadeIn 0.5s ease-in,glow 2s infinite;}
    .robot-container{position:relative;width:100px;height:100px;margin:20px auto;}
    .robot{width:80px;height:80px;background:linear-gradient(135deg,#00d4ff,#007acc);border-radius:50%;position:relative;animation:bounce 1s infinite;}
    .robot-eye{width:20px;height:20px;background:#fff;border-radius:50%;position:absolute;top:20px;left:20px;animation:blink 2s infinite;}
    .robot-eye.right{left:40px;}
    @keyframes bounce{0%,100%{transform:translateY(0);}50%{transform:translateY(-10px);}}
    @keyframes blink{0%,100%{height:20px;}10%{height:5px;}}
    @keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
    @keyframes glow{0%{box-shadow:0 0 5px #00d4ff;}50%{box-shadow:0 0 20px #00d4ff;}100%{box-shadow:0 0 5px #00d4ff;}}
    .stButton>button{background:linear-gradient(90deg,#ff6b6b,#ff8787);color:white;padding:12px 25px;border-radius:8px;margin:5px;box-shadow:0 4px 8px rgba(0,0,0,0.3);transition:transform 0.2s ease,box-shadow 0.2s ease;}
    .stButton>button:hover{background:linear-gradient(90deg,#ff8787,#ff6b6b);transform:translateY(-2px);box-shadow:0 6px 12px rgba(0,0,0,0.4);}
    .stTextInput>div>input{border:2px solid #00d4ff;border-radius:8px;padding:10px;font-size:1.1em;background:#0f172a;color:#e2e8f0;box-shadow:inset 0 0 5px rgba(0,212,255,0.2);}
    .footer{text-align:center;padding:20px;background:linear-gradient(145deg,#1a1a2e,#16213e);border-top:2px solid #00d4ff;font-size:1em;color:#b0bec5;position:fixed;bottom:0;width:100%;box-shadow:0 -4px 8px rgba(0,0,0,0.5);}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron&family=Roboto+Mono&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Authentication Setup
credentials = {
    "usernames": {
        "free_user": {"name": "Free User", "password": stauth.Hasher(["free123"]).generate()[0], "tier": "free"},
        "premium_user": {"name": "Premium User", "password": stauth.Hasher(["premium456"]).generate()[0], "tier": "premium"}
    }
}
authenticator = stauth.Authenticate(credentials, "ai_robot_app", "auth_key", cookie_expiry_days=30)

# Gemini API Config
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    st.error(f"⚠️ API Key Error: {e}")
    st.stop()

# Custom LLM Wrapper
class GeminiLLM(LLM):
    def _call(self, prompt: str, stop=None):
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, "text") else "Error: No response!"

    def predict(self, prompt: str):
        return self._call(prompt)

    @property
    def _llm_type(self):
        return "Gemini"

# Initialize Components
llm = GeminiLLM()
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    You are a futuristic AI Data Science Robot Tutor! Engage users with enthusiasm, provide clear, concise data science answers, and add a robotic flair (e.g., "Beep boop! Analyzing..."). Use examples and stay on topic!
    History: {history}
    User: {input}
    AI Robot: Beep boop! Processing your query... Here’s your data science intel:
    """
)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, prompt=prompt_template, memory=memory)

# Text-to-Speech (Cached for Efficiency)
@st.cache_data
def text_to_speech(text):
    try:
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in ".,!?")
        if not clean_text.strip():
            clean_text = "Beep boop! No data to vocalize!"
        tts = gTTS(text=clean_text[:500], lang='en', slow=False)  # Limit for efficiency
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except Exception as e:
        st.warning(f"Voice error: {e}")
        return None

# Robot and Spline Components (unchanged for brevity)
def render_robot():
    components.html(
        """
        <div class="robot-container">
            <div class="robot">
                <div class="robot-eye"></div>
                <div class="robot-eye right"></div>
            </div>
        </div>
        <script>
            const robot = document.querySelector('.robot');
            robot.addEventListener('click', () => {
                robot.style.animation = 'bounce 0.5s 2';
                setTimeout(() => { robot.style.animation = 'bounce 1s infinite'; }, 1000);
            });
        </script>
        """,
        height=120
    )

def render_spline_scene():
    components.html(
        """
        <script type="module">
            import { Application } from 'https://unpkg.com/@splinetool/runtime@latest/build/runtime.js';
            const canvas = document.createElement('canvas');
            canvas.style.width = '100%';
            canvas.style.height = '300px'; // Reduced height for performance
            document.getElementById('spline-scene').appendChild(canvas);
            const spline = new Application(canvas);
            spline.load('https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode');
        </script>
        <div id="spline-scene"></div>
        """,
        height=300,
    )

# Main App with Authentication
def main():
    name, authentication_status, username = authenticator.login("Login", "main")
    
    if authentication_status:
        st.markdown('<div class="main-title">AI Data Science Robot</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="subtitle">🧠 Welcome, {name}! Your Virtual AI Mentor Awaits!</div>', unsafe_allow_html=True)

        # Subscription Check
        user_tier = credentials["usernames"][username]["tier"]
        query_limit = 5 if user_tier == "free" else None
        if "query_count" not in st.session_state:
            st.session_state.query_count = 0

        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Your Cyber Assistant")
            render_robot()

            st.markdown("### Chat Interface")
            chat_container = st.empty()
            chat_html = ""
            if "history" not in st.session_state:
                st.session_state.history = [{"user": "", "ai": f"Beep boop! Greetings, {name}! I’m your Data Science Robot—ready to assist. What’s your query?"}]
                st.session_state.last_audio = text_to_speech(st.session_state.history[0]["ai"])
            for exchange in st.session_state.history:
                if exchange["user"]:
                    chat_html += f'<div class="user-msg"><b>{name}:</b> {exchange["user"]}</div>'
                if exchange["ai"]:
                    chat_html += f'<div class="ai-msg"><b>Robot:</b> {exchange["ai"]}</div>'
            chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
            chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)

        with col2:
            with st.form(key="input_form", clear_on_submit=True):
                user_input = st.text_input("Input Query!", placeholder="E.g., Explain neural networks")
                submit_button = st.form_submit_button(label="Transmit!")

            if submit_button and user_input.strip():
                if user_tier == "free" and st.session_state.query_count >= query_limit:
                    st.warning("Beep boop! Free tier limit reached. Upgrade to Premium for unlimited queries!")
                    st.markdown('<a href="https://buy.stripe.com/test_123" target="_blank">Upgrade Now</a>', unsafe_allow_html=True)
                else:
                    with st.spinner("Beep boop! Computing..."):
                        response = conversation.run(input=user_input)
                        st.session_state.history.append({"user": user_input, "ai": response})
                        chat_html += f'<div class="user-msg"><b>{name}:</b> {user_input}</div>'
                        chat_html += f'<div class="ai-msg"><b>Robot:</b> {response}</div>'
                        chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
                        chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
                        st.session_state.last_audio = text_to_speech(response)
                        if st.session_state.last_audio:
                            st.audio(st.session_state.last_audio, format="audio/mp3")
                        st.session_state.query_count += 1
            elif submit_button:
                st.warning("Beep! Input required!")

            st.markdown("### Cyber Tools")
            col3, col4, col5 = st.columns(3)
            with col3:
                if st.button("Random Data Bit"):
                    tip = conversation.run("Transmit a quick data science fact!")
                    st.session_state.history.append({"user": "Random data bit", "ai": tip})
                    chat_html += f'<div class="user-msg"><b>{name}:</b> Random data bit</div>'
                    chat_html += f'<div class="ai-msg"><b>Robot:</b> {tip}</div>'
                    chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
            with col4:
                if st.button("Replay Signal"):
                    if st.session_state.get("last_audio"):
                        st.audio(st.session_state.last_audio, format="audio/mp3")
            with col5:
                if st.button("Reset Core"):
                    st.session_state.history = [{"user": "", "ai": "Beep boop! Core reset—ready for new data!"}]
                    st.session_state.query_count = 0

            if user_tier == "premium":
                st.markdown("### Premium Feature: 3D Scene")
                render_spline_scene()

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="footer">Built with ❤️ by Gopichand | Tier: {user_tier.capitalize()}</div>', unsafe_allow_html=True)
        authenticator.logout("Logout", "sidebar")
    elif authentication_status is False:
        st.error("Username/password incorrect!")
    elif authentication_status is None:
        st.warning("Please enter your credentials to access the AI Robot!")

if __name__ == "__main__":
    main()
