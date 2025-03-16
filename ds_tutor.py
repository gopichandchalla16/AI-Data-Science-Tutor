import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from gtts import gTTS
import io
import streamlit.components.v1 as components

# Streamlit Page Config
st.set_page_config(page_title="AI Data Science Tutor", page_icon="🤖", layout="wide")

# Add CSS Instructions (Enhanced for Custom UI)
st.markdown("""
    <style>
    .main-title { 
        font-size: 4em; 
        color: #00d4ff; 
        text-align: center; 
        font-weight: bold; 
        font-family: 'Orbitron', sans-serif; 
        text-shadow: 0 0 15px #00d4ff, 0 0 30px #00d4ff; 
        margin-bottom: 5px; 
    }
    .subtitle { 
        text-align: center; 
        color: #b0bec5; 
        font-size: 1.5em; 
        font-family: 'Roboto Mono', monospace; 
        text-shadow: 0 0 5px #b0bec5; 
        margin-bottom: 20px; 
    }
    .card { 
        border-radius: 20px; 
        background: linear-gradient(145deg, #1a1a2e, #16213e); 
        padding: 30px; 
        box-shadow: 0 10px 20px rgba(0,0,0,0.6), inset 0 0 15px rgba(0,212,255,0.3); 
        position: relative; 
        overflow: hidden; 
        transition: transform 0.3s ease; 
    }
    .card:hover { 
        transform: scale(1.02); 
    }
    .spotlight { 
        position: absolute; 
        top: -60px; 
        left: -60px; 
        width: 250px; 
        height: 250px; 
        background: radial-gradient(circle, rgba(0,212,255,0.5), transparent 70%); 
        animation: pulse 5s infinite; 
        z-index: 0; 
    }
    .chat-box { 
        border: 4px solid #00d4ff; 
        border-radius: 15px; 
        padding: 25px; 
        background: linear-gradient(135deg, #0f172a, #1e293b); 
        box-shadow: 0 6px 15px rgba(0,0,0,0.6), inset 0 0 10px rgba(0,212,255,0.2); 
        min-height: 500px; 
        max-height: 500px; 
        overflow-y: auto; 
        font-size: 1.2em; 
    }
    .user-msg { 
        background: #334155; 
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0; 
        color: #e2e8f0; 
        box-shadow: 0 3px 6px rgba(0,0,0,0.4); 
        font-family: 'Roboto Mono', monospace; 
    }
    .ai-msg { 
        background: linear-gradient(90deg, #00d4ff, #007acc); 
        color: #fff; 
        padding: 18px; 
        border-radius: 12px; 
        margin: 10px 0; 
        font-weight: bold; 
        box-shadow: 0 5px 10px rgba(0,0,0,0.5); 
        animation: fadeIn 0.5s ease-in, glow 2s infinite; 
        font-family: 'Orbitron', sans-serif; 
    }
    .tool-card { 
        border-radius: 10px; 
        background: linear-gradient(145deg, #1e293b, #0f172a); 
        padding: 15px; 
        text-align: center; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.4); 
        transition: transform 0.2s ease, box-shadow 0.2s ease; 
        color: #b0bec5; 
        font-family: 'Roboto Mono', monospace; 
    }
    .tool-card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 6px 12px rgba(0,212,255,0.3); 
        color: #00d4ff; 
    }
    .input-bar { 
        border: 3px solid #00d4ff; 
        border-radius: 10px; 
        padding: 15px; 
        background: #0f172a; 
        box-shadow: inset 0 0 10px rgba(0,212,255,0.2); 
        font-size: 1.2em; 
        color: #e2e8f0; 
        font-family: 'Roboto Mono', monospace; 
    }
    .stButton>button { 
        background: linear-gradient(90deg, #ff6b6b, #ff8787); 
        color: white; 
        font-size: 18px; 
        padding: 15px 30px; 
        border-radius: 10px; 
        box-shadow: 0 5px 10px rgba(0,0,0,0.4); 
        transition: transform 0.2s ease, box-shadow 0.2s ease; 
        font-family: 'Orbitron', sans-serif; 
    }
    .stButton>button:hover { 
        background: linear-gradient(90deg, #ff8787, #ff6b6b); 
        transform: translateY(-3px); 
        box-shadow: 0 8px 15px rgba(0,0,0,0.5); 
    }
    .footer { 
        text-align: center; 
        padding: 15px; 
        background: linear-gradient(145deg, #1a1a2e, #16213e); 
        border-top: 2px solid #00d4ff; 
        font-size: 1em; 
        color: #b0bec5; 
        position: fixed; 
        bottom: 0; 
        width: 100%; 
        box-shadow: 0 -4px 8px rgba(0,0,0,0.5); 
    }
    @keyframes fadeIn { 
        from { opacity: 0; transform: translateY(10px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    @keyframes glow { 
        0% { box-shadow: 0 0 5px #00d4ff; } 
        50% { box-shadow: 0 0 20px #00d4ff; } 
        100% { box-shadow: 0 0 5px #00d4ff; } 
    }
    @keyframes pulse { 
        0% { transform: scale(1); opacity: 0.5; } 
        50% { transform: scale(1.3); opacity: 0.7; } 
        100% { transform: scale(1); opacity: 0.5; } 
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
    You are a futuristic AI Data Science Robot Tutor! Engage users with enthusiasm, provide clear, concise data science answers, and add a robotic flair (e.g., "Beep boop! Analyzing..."). Use examples and stay on topic!

    History:
    {history}

    User: {input}
    AI Robot: Beep boop! Processing your query... Here’s your data science intel:
    """
)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, prompt=prompt_template, memory=memory)

# Text-to-Speech Function
def text_to_speech(text):
    try:
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in ".,!?")
        if not clean_text.strip():
            clean_text = "Beep boop! No data to vocalize—input required!"
        tts = gTTS(text=clean_text, lang='en', slow=False, tld='co.uk')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except Exception as e:
        st.warning(f"Voice module error: {str(e)}")
        return None

# Embed the 3D Spline Scene
def render_spline_scene():
    components.html(
        """
        <script type="module">
            import { Application } from 'https://unpkg.com/@splinetool/runtime@latest/build/runtime.js';
            const canvas = document.createElement('canvas');
            canvas.style.width = '100%';
            canvas.style.height = '400px';
            canvas.style.borderRadius = '15px';
            canvas.style.boxShadow = '0 5px 15px rgba(0,212,255,0.2)';
            document.getElementById('spline-scene').appendChild(canvas);
            const spline = new Application(canvas);
            spline.load('https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode');
        </script>
        <div id="spline-scene"></div>
        """,
        height=410
    )

# Main App
def main():
    # Header with Animated Robot
    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #00d4ff, #007acc); border-radius: 50%; animation: bounce 1.5s infinite;">
                <div style="width: 15px; height: 15px; background: #fff; border-radius: 50%; position: absolute; top: 15px; left: 15px; animation: blink 2s infinite;"></div>
                <div style="width: 15px; height: 15px; background: #fff; border-radius: 50%; position: absolute; top: 15px; left: 30px; animation: blink 2s infinite;"></div>
            </div>
            <div class="main-title">AI Data Science Tutor</div>
        </div>
        <div class="subtitle">🤖 Your Galactic Guide to Data Science Mastery</div>
    """, unsafe_allow_html=True)

    # Main Layout with Two Columns
    col1, col2 = st.columns([2, 1])

    # Left Column: Chat Interface
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="spotlight"></div>', unsafe_allow_html=True)
        st.markdown("### Cyber Chat Core", unsafe_allow_html=True)
        with st.container():
            chat_container = st.empty()
            chat_html = ""
            if "history" not in st.session_state:
                st.session_state.history = [{"user": "", "ai": "Beep boop! Greetings, human! I’m your Data Science Tutor—ready to compute. What’s your query?"}]
                st.session_state.last_audio = text_to_speech(st.session_state.history[0]["ai"])
            for exchange in st.session_state.history:
                if exchange["user"]:
                    chat_html += f'<div class="user-msg"><b>You:</b> {exchange["user"]}</div>'
                if exchange["ai"]:
                    chat_html += f'<div class="ai-msg"><b>Tutor:</b> {exchange["ai"]}</div>'
            chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
            chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Right Column: Input and Tools
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="spotlight"></div>', unsafe_allow_html=True)

        # Input Section
        st.markdown("### Transmit Query", unsafe_allow_html=True)
        with st.form(key="input_form", clear_on_submit=True):
            user_input = st.text_input("", placeholder="Ask me anything about Data Science!", key="input", label_visibility="collapsed")
            submit_button = st.form_submit_button(label="Transmit 🚀")

        if submit_button and user_input.strip():
            with st.spinner("Beep boop! Computing..."):
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
            st.warning("Beep! Input required for processing!")

        # Toolbox Section
        st.markdown("### Cyber Toolbox", unsafe_allow_html=True)
        tool_col1, tool_col2, tool_col3 = st.columns(3)
        with tool_col1:
            st.markdown('<div class="tool-card">📊 Random Data Bit</div>', unsafe_allow_html=True)
            if st.button("Generate", key="tip_btn"):
                tip = conversation.run("Transmit a quick data science fact!")
                st.session_state.history.append({"user": "Random data bit", "ai": tip})
                chat_html += f'<div class="user-msg"><b>You:</b> Random data bit</div>'
                chat_html += f'<div class="ai-msg"><b>Tutor:</b> {tip}</div>'
                chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
                st.session_state.last_audio = text_to_speech(tip)
                if st.session_state.last_audio:
                    st.audio(st.session_state.last_audio, format="audio/mp3")
        with tool_col2:
            st.markdown('<div class="tool-card">🔊 Replay Signal</div>', unsafe_allow_html=True)
            if st.button("Replay", key="repeat_btn"):
                if st.session_state.get("last_audio"):
                    st.audio(st.session_state.last_audio, format="audio/mp3")
                    st.success("Beep! Signal replayed!")
        with tool_col3:
            st.markdown('<div class="tool-card">🔄 Reset Core</div>', unsafe_allow_html=True)
            if st.button("Reset", key="clear_btn"):
                st.session_state.history = [{"user": "", "ai": "Beep boop! Core reset—ready for new data!"}]
                st.session_state.last_audio = text_to_speech("Beep boop! Core reset—ready for new data!")
                chat_container.markdown('<div class="chat-box"><div class="ai-msg"><b>Tutor:</b> Beep boop! Core reset—ready for new data!</div></div>', unsafe_allow_html=True)

        # Spline Scene
        st.markdown("### 3D Data Visualizer", unsafe_allow_html=True)
        render_spline_scene()

        st.markdown('</div>', unsafe_allow_html=True)

    # Simple Footer with Gopi Chand's Name
    st.markdown("""
        <div class="footer">
            <p style="margin: 0; font-family: 'Roboto Mono', monospace; color: #b0bec5; font-size: 0.9em;">
                Created by <b>Gopi Chand</b> | AI Data Science Tutor © 2025
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
