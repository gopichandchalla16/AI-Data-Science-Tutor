import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from gtts import gTTS
import io
from datetime import datetime
import streamlit.components.v1 as components

# Streamlit Page Config
st.set_page_config(page_title="AI Data Science Nexus", page_icon="🌌", layout="wide")

# Ultra-Creative CSS
st.markdown("""
    <style>
    body { 
        background: linear-gradient(135deg, #0d1321, #1a1a2e); 
        overflow: hidden; 
    }
    .cyber-grid { 
        position: fixed; 
        top: 0; 
        left: 0; 
        width: 100%; 
        height: 100%; 
        background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMCwwIEgyMDBNMjAwLDAgVjIwME0wLDIwMCBIMjAwTTAyMDAgVjAiIGZpbGw9Im5vbmUiIHN0cm9rZT0icmdiYSgwLDIyOCwyNTUsMC4xKSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9zdmc+'); 
        background-size: 50px 50px; 
        opacity: 0.3; 
        z-index: -1; 
        animation: gridPulse 10s infinite; 
    }
    .main-title { 
        font-size: 4em; 
        color: #00f4ff; 
        text-align: center; 
        font-family: 'Orbitron', sans-serif; 
        text-shadow: 0 0 20px #00f4ff, 0 0 40px #00f4ff; 
        animation: neonFlicker 3s infinite; 
        margin-top: 20px; 
    }
    .subtitle { 
        text-align: center; 
        color: #80e8ff; 
        font-size: 1.5em; 
        font-family: 'Roboto Mono', monospace; 
        text-shadow: 0 0 10px #80e8ff; 
        margin-bottom: 30px; 
    }
    .card { 
        border-radius: 25px; 
        background: rgba(26, 42, 62, 0.9); 
        padding: 25px; 
        box-shadow: 0 15px 40px rgba(0,0,0,0.7), inset 0 0 20px rgba(0,244,255,0.3); 
        backdrop-filter: blur(5px); 
        transition: transform 0.4s ease, box-shadow 0.4s ease; 
    }
    .card:hover { 
        transform: translateY(-10px); 
        box-shadow: 0 20px 50px rgba(0,0,0,0.9); 
    }
    .chat-box { 
        border: 2px dashed #00f4ff; 
        border-radius: 15px; 
        padding: 20px; 
        background: rgba(15, 23, 42, 0.8); 
        min-height: 500px; 
        max-height: 500px; 
        overflow-y: auto; 
        font-family: 'Roboto Mono', monospace; 
        box-shadow: inset 0 0 15px rgba(0,244,255,0.2); 
        animation: borderGlow 2s infinite; 
    }
    .user-msg { 
        background: linear-gradient(135deg, #2d4266, #1e2a3d); 
        padding: 12px 18px; 
        border-radius: 10px; 
        margin: 12px 0; 
        color: #d1e8ff; 
        box-shadow: 0 3px 8px rgba(0,0,0,0.4); 
        transition: transform 0.2s ease; 
    }
    .user-msg:hover { transform: scale(1.02); }
    .ai-msg { 
        background: linear-gradient(90deg, #00f4ff, #1a73e8); 
        color: #fff; 
        padding: 15px 20px; 
        border-radius: 10px; 
        margin: 12px 0; 
        font-weight: bold; 
        box-shadow: 0 5px 15px rgba(0,244,255,0.5); 
        animation: fadeIn 0.6s ease-in; 
    }
    /* Hyper-Creative Robot */
    .robot-container { 
        width: 150px; 
        height: 180px; 
        margin: 20px auto; 
        position: relative; 
    }
    .robot { 
        width: 120px; 
        height: 120px; 
        background: linear-gradient(135deg, #00f4ff, #1a73e8); 
        border-radius: 30px 30px 60% 60%; 
        position: absolute; 
        top: 20px; 
        left: 15px; 
        box-shadow: 0 0 30px rgba(0,244,255,0.9), inset 0 0 15px rgba(255,255,255,0.3); 
        animation: hoverGlow 2s infinite ease-in-out; 
        transition: transform 0.3s ease; 
    }
    .robot:hover { transform: scale(1.15) rotate(5deg); }
    .robot-core { 
        width: 40px; 
        height: 40px; 
        background: radial-gradient(circle, #ff4081, transparent); 
        border-radius: 50%; 
        position: absolute; 
        top: 50%; 
        left: 50%; 
        transform: translate(-50%, -50%); 
        animation: corePulse 1.5s infinite; 
    }
    .robot-eye { 
        width: 20px; 
        height: 20px; 
        background: #ffdd00; 
        border-radius: 50%; 
        position: absolute; 
        top: 30px; 
        left: 35px; 
        box-shadow: 0 0 15px #ffdd00; 
        animation: eyeGlow 2s infinite; 
    }
    .robot-eye.right { left: 65px; }
    .data-orbits { 
        position: absolute; 
        width: 140px; 
        height: 140px; 
        top: 10px; 
        left: 5px; 
        border: 2px dotted rgba(0,244,255,0.5); 
        border-radius: 50%; 
        animation: orbitSpin 6s infinite linear; 
    }
    @keyframes hoverGlow { 
        0%, 100% { transform: translateY(0); box-shadow: 0 0 30px rgba(0,244,255,0.9); } 
        50% { transform: translateY(-15px); box-shadow: 0 0 50px rgba(0,244,255,1); } 
    }
    @keyframes corePulse { 
        0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.8; } 
        50% { transform: translate(-50%, -50%) scale(1.2); opacity: 1; } 
    }
    @keyframes eyeGlow { 
        0%, 100% { box-shadow: 0 0 15px #ffdd00; } 
        50% { box-shadow: 0 0 25px #ffdd00; } 
    }
    @keyframes orbitSpin { 
        0% { transform: rotate(0deg); } 
        100% { transform: rotate(360deg); } 
    }
    @keyframes fadeIn { 
        from { opacity: 0; transform: translateY(15px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    @keyframes neonFlicker { 
        0%, 100% { opacity: 1; } 
        50% { opacity: 0.95; } 
    }
    @keyframes gridPulse { 
        0%, 100% { opacity: 0.3; } 
        50% { opacity: 0.5; } 
    }
    @keyframes borderGlow { 
        0%, 100% { border-color: #00f4ff; } 
        50% { border-color: #1a73e8; } 
    }
    .stButton>button { 
        background: linear-gradient(90deg, #ff4081, #ff6b6b); 
        color: #fff; 
        padding: 12px 25px; 
        border-radius: 10px; 
        font-family: 'Orbitron', sans-serif; 
        box-shadow: 0 5px 15px rgba(255,64,129,0.5); 
        transition: all 0.3s ease; 
    }
    .stButton>button:hover { 
        background: linear-gradient(90deg, #ff6b6b, #ff4081); 
        transform: scale(1.05); 
        box-shadow: 0 8px 20px rgba(255,64,129,0.7); 
    }
    .stTextInput>div>input { 
        border: 2px solid #00f4ff; 
        border-radius: 10px; 
        padding: 15px; 
        background: rgba(26,42,62,0.8); 
        color: #d1e8ff; 
        font-family: 'Roboto Mono', monospace; 
        box-shadow: inset 0 0 10px rgba(0,244,255,0.3); 
        transition: border-color 0.3s ease; 
    }
    .stTextInput>div>input:focus { border-color: #ff4081; }
    .footer { 
        text-align: center; 
        padding: 20px; 
        background: rgba(22,33,62,0.9); 
        color: #80e8ff; 
        font-family: 'Roboto Mono', monospace; 
        position: fixed; 
        bottom: 0; 
        width: 100%; 
        text-shadow: 0 0 5px #80e8ff; 
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron&family=Roboto+Mono&display=swap" rel="stylesheet">
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
            return response.text if response and hasattr(response, "text") else "Error: No response!"
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
    You are a cosmic AI Data Science Nexus! Deliver concise, insightful answers with a futuristic flair (e.g., "Beep zap! Processing..."). Inspire users with examples and stay on topic!

    History:
    {history}

    User: {input}
    AI Nexus: Beep zap! Processing your query... Here’s your cosmic data insight:
    """
)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, prompt=prompt_template, memory=memory)

# Text-to-Speech Function
def text_to_speech(text):
    try:
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in ".,!?")
        if not clean_text.strip():
            clean_text = "Beep zap! Query required—engage me!"
        tts = gTTS(text=clean_text, lang='en', slow=False, tld='co.uk')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except Exception as e:
        st.warning(f"Nexus voice error: {str(e)}")
        return None

# Cosmic Robot with Orbits
def render_robot():
    components.html(
        """
        <div class="robot-container">
            <div class="data-orbits"></div>
            <div class="robot" title="Interact with me!">
                <div class="robot-core"></div>
                <div class="robot-eye"></div>
                <div class="robot-eye right"></div>
            </div>
        </div>
        <script>
            const robot = document.querySelector('.robot');
            robot.addEventListener('click', () => {
                robot.style.animation = 'hoverGlow 0.5s 2';
                setTimeout(() => { robot.style.animation = 'hoverGlow 2s infinite ease-in-out'; }, 1000);
            });
        </script>
        """,
        height=200
    )

# Main App
def main():
    st.markdown('<div class="cyber-grid"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Data Science Nexus</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🌌 Your Cosmic AI Guide to Infinite Knowledge</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.3])

        # Left: Robot and Chat
        with col1:
            st.markdown("### 🌠 Nexus Core", help="Your cosmic companion awaits!")
            render_robot()

            st.markdown("#### Transmission Log")
            chat_container = st.empty()
            chat_html = ""
            if "history" not in st.session_state:
                st.session_state.history = [{"user": "", "ai": "Beep zap! Greetings, explorer! I’m your Data Science Nexus. What knowledge do you seek?"}]
                st.session_state.last_audio = text_to_speech(st.session_state.history[0]["ai"])
            for exchange in st.session_state.history:
                if exchange["user"]:
                    chat_html += f'<div class="user-msg"><b>Explorer:</b> {exchange["user"]}</div>'
                if exchange["ai"]:
                    chat_html += f'<div class="ai-msg"><b>Nexus:</b> {exchange["ai"]}</div>'
            chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
            chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)

        # Right: Input and Controls
        with col2:
            st.markdown("### Query the Cosmos")
            with st.form(key="input_form", clear_on_submit=True):
                user_input = st.text_input("", placeholder="Ask me anything—e.g., What’s a neural net?", label_visibility="collapsed")
                submit_button = st.form_submit_button(label="🌠 Transmit")

            if submit_button and user_input.strip():
                with st.spinner("Beep zap! Decoding..."):
                    response = conversation.run(input=user_input)
                    st.session_state.history.append({"user": user_input, "ai": response})
                    chat_html += f'<div class="user-msg"><b>Explorer:</b> {exchange["user"]}</div>'
                    chat_html += f'<div class="ai-msg"><b>Nexus:</b> {response}</div>'
                    chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
                    chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
                    st.session_state.last_audio = text_to_speech(response)
                    if st.session_state.last_audio:
                        st.audio(st.session_state.last_audio, format="audio/mp3")
            elif submit_button:
                st.warning("Beep! Input required to engage the Nexus.")

            st.markdown("### Cosmic Tools")
            col3, col4, col5 = st.columns(3)
            with col3:
                if st.button("✨ Data Spark", help="Ignite a random insight"):
                    tip = conversation.run("Share a cosmic data science fact!")
                    st.session_state.history.append({"user": "Data spark", "ai": tip})
                    chat_html += f'<div class="user-msg"><b>Explorer:</b> Data spark</div>'
                    chat_html += f'<div class="ai-msg"><b>Nexus:</b> {tip}</div>'
                    chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
                    st.session_state.last_audio = text_to_speech(tip)
                    if st.session_state.last_audio:
                        st.audio(st.session_state.last_audio, format="audio/mp3")
            with col4:
                if st.button("🔊 Echo", help="Replay the last transmission"):
                    if st.session_state.get("last_audio"):
                        st.audio(st.session_state.last_audio, format="audio/mp3")
                        st.success("Beep zap! Echoed!")
            with col5:
                if st.button("🌌 Reboot", help="Reset the Nexus core"):
                    st.session_state.history = [{"user": "", "ai": "Beep zap! Core rebooted—ready for exploration!"}]
                    st.session_state.last_audio = text_to_speech("Beep zap! Core rebooted—ready for exploration!")
                    chat_container.markdown('<div class="chat-box"><div class="ai-msg"><b>Nexus:</b> Beep zap! Core rebooted—ready for exploration!</div></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Crafted by Gopichand | Powered by Cosmic AI</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
