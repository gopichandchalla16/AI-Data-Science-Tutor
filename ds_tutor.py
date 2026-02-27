import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import io
import streamlit.components.v1 as components

# -----------------------
# Streamlit Page Config
# -----------------------
st.set_page_config(page_title="AI Data Science Robot", page_icon="🤖", layout="wide")

# -----------------------
# Custom CSS Styling
# -----------------------
st.markdown("""
<style>
.main-title { 
    font-size: 3.5em; 
    color: #00d4ff; 
    text-align: center; 
    font-weight: bold; 
    font-family: 'Orbitron', sans-serif; 
    text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff; 
}
.subtitle { 
    text-align: center; 
    color: #b0bec5; 
    font-size: 1.3em; 
    font-family: 'Roboto Mono', monospace; 
}
.chat-box { 
    border: 3px solid #00d4ff; 
    border-radius: 12px; 
    padding: 20px; 
    background: #0f172a; 
    min-height: 400px; 
    overflow-y: auto; 
}
.user-msg { 
    background: #334155; 
    padding: 12px; 
    border-radius: 8px; 
    margin: 8px 0; 
    color: #e2e8f0; 
}
.ai-msg { 
    background: linear-gradient(90deg, #00d4ff, #007acc); 
    color: white; 
    padding: 15px; 
    border-radius: 10px; 
    margin: 8px 0; 
    font-weight: bold; 
}
.footer { 
    text-align: center; 
    padding: 20px; 
    font-size: 0.9em; 
    color: #b0bec5; 
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Orbitron&family=Roboto+Mono&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# -----------------------
# Gemini API Setup
# -----------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Please set GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro")

# -----------------------
# Session Memory
# -----------------------
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = [
        {
            "user": "",
            "ai": "Beep boop. Greetings human. I am your AI Data Science Robot. Ask me anything about statistics, machine learning or Python."
        }
    ]

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# -----------------------
# Text to Speech
# -----------------------
def text_to_speech(text):
    try:
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in ".,!?")
        tts = gTTS(text=clean_text, lang="en")
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except:
        return None

# -----------------------
# Generate AI Response
# -----------------------
def generate_response(user_input):
    conversation_history = ""
    for msg in st.session_state.chat_memory:
        conversation_history += f"User: {msg['user']}\nAI: {msg['ai']}\n"

    prompt = f"""
You are a futuristic AI Data Science Robot Tutor.
Give clear and concise data science explanations.
Add light robotic personality occasionally.

Conversation so far:
{conversation_history}

User: {user_input}
AI:
"""

    try:
        response_obj = model.generate_content(prompt)
        response = response_obj.text if hasattr(response_obj, "text") else "Error generating response."
    except:
        response = "Error generating response."

    st.session_state.chat_memory.append({"user": user_input, "ai": response})
    return response

# -----------------------
# Spline 3D Scene
# -----------------------
def render_spline_scene():
    components.html("""
        <script type="module">
            import { Application } from 'https://unpkg.com/@splinetool/runtime@latest/build/runtime.js';
            const canvas = document.createElement('canvas');
            canvas.style.width = '100%';
            canvas.style.height = '500px';
            document.getElementById('spline-scene').appendChild(canvas);
            const spline = new Application(canvas);
            spline.load('https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode');
        </script>
        <div id="spline-scene"></div>
    """, height=500)

# -----------------------
# UI Layout
# -----------------------
def main():
    st.markdown('<div class="main-title">AI Data Science Robot</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your Virtual AI Mentor</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Chat Interface")

        chat_html = ""
        for exchange in st.session_state.chat_memory:
            if exchange["user"]:
                chat_html += f'<div class="user-msg"><b>You:</b> {exchange["user"]}</div>'
            chat_html += f'<div class="ai-msg"><b>Robot:</b> {exchange["ai"]}</div>'

        st.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)

    with col2:
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask your question")
            submit = st.form_submit_button("Transmit")

        if submit and user_input.strip():
            response = generate_response(user_input)
            audio = text_to_speech(response)
            if audio:
                st.session_state.last_audio = audio
                st.audio(audio, format="audio/mp3")
            st.rerun()

        st.markdown("### Tools")

        if st.button("Random Data Fact"):
            fact = generate_response("Give me a short interesting data science fact.")
            st.rerun()

        if st.button("Replay Voice"):
            if st.session_state.last_audio:
                st.audio(st.session_state.last_audio, format="audio/mp3")

        if st.button("Reset Chat"):
            st.session_state.chat_memory = []
            st.session_state.last_audio = None
            st.rerun()

        st.markdown("### Interactive 3D Scene")
        render_spline_scene()

    st.markdown('<div class="footer">Created by GopiChand | AI Data Science Tutor 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
