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
st.set_page_config(page_title="AI Data Science Tutor", page_icon="🤖", layout="wide")

# Enhanced CSS with Improved Aesthetics
st.markdown("""
    <style>
    body { background: #0f172a; }
    .main-title { 
        font-size: 3.5em; 
        color: #00e4ff; 
        text-align: center; 
        font-family: 'Orbitron', sans-serif; 
        text-shadow: 0 0 15px #00e4ff; 
        margin-top: 10px; 
    }
    .subtitle { 
        text-align: center; 
        color: #b0bec5; 
        font-size: 1.2em; 
        font-family: 'Roboto Mono', monospace; 
        text-shadow: 0 0 5px #b0bec5; 
        margin-bottom: 20px; 
    }
    .card { 
        border-radius: 20px; 
        background: linear-gradient(145deg, #1e293b, #16213e); 
        padding: 20px; 
        box-shadow: 0 10px 20px rgba(0,0,0,0.6), inset 0 0 15px rgba(0,228,255,0.2); 
        transition: transform 0.3s ease, box-shadow 0.3s ease; 
    }
    .card:hover { 
        transform: scale(1.02); 
        box-shadow: 0 15px 30px rgba(0,0,0,0.8); 
    }
    .chat-box { 
        border: 2px solid #00e4ff; 
        border-radius: 10px; 
        padding: 15px; 
        background: #0f172a; 
        box-shadow: inset 0 0 10px rgba(0,228,255,0.1); 
        min-height: 450px; 
        max-height: 450px; 
        overflow-y: auto; 
        font-family: 'Roboto Mono', monospace; 
    }
    .user-msg { 
        background: #334155; 
        padding: 10px 15px; 
        border-radius: 8px; 
        margin: 10px 0; 
        color: #e2e8f0; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.3); 
    }
    .ai-msg { 
        background: linear-gradient(90deg, #00e4ff, #1a73e8); 
        color: #fff; 
        padding: 12px 18px; 
        border-radius: 8px; 
        margin: 10px 0; 
        font-weight: bold; 
        box-shadow: 0 3px 8px rgba(0,0,0,0.4); 
        animation: fadeIn 0.5s ease-in; 
    }
    /* Enhanced Robot Styling */
    .robot-container { 
        width: 120px; 
        height: 120px; 
        margin: 20px auto; 
        position: relative; 
    }
    .robot { 
        width: 100px; 
        height: 100px; 
        background: linear-gradient(135deg, #00e4ff, #1a73e8); 
        border-radius: 20px 20px 50% 50%; 
        position: absolute; 
        top: 10px; 
        left: 10px; 
        box-shadow: 0 0 20px rgba(0,228,255,0.8); 
        animation: hover 2s infinite ease-in-out; 
        transition: transform 0.2s ease; 
    }
    .robot:hover { transform: scale(1.1); }
    .robot-eye { 
        width: 15px; 
        height: 15px; 
        background: #ff4081; 
        border-radius: 50%; 
        position: absolute; 
        top: 30px; 
        left: 30px; 
        box-shadow: 0 0 10px #ff4081; 
        animation: glow-eye 1.5s infinite; 
    }
    .robot-eye.right { left: 55px; }
    .data-stream { 
        position: absolute; 
        bottom: 10px; 
        left: 10px; 
        width: 80px; 
        height: 10px; 
        background: rgba(0,228,255,0.2); 
        border-radius: 5px; 
        animation: data-flow 2s infinite linear; 
    }
    @keyframes hover { 
        0%, 100% { transform: translateY(0); } 
        50% { transform: translateY(-10px); } 
    }
    @keyframes glow-eye { 
        0%, 100% { box-shadow: 0 0 10px #ff4081; } 
        50% { box-shadow: 0 0 15px #ff4081; } 
    }
    @keyframes data-flow { 
        0% { background-position: 0%; } 
        100% { background-position: 200%; } 
    }
    @keyframes fadeIn { 
        from { opacity: 0; transform: translateY(10px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    .stButton>button { 
        background: linear-gradient(90deg, #ff6b6b, #ff8787); 
        color: white; 
        padding: 10px 20px; 
        border-radius: 8px; 
        font-family: 'Roboto Mono', monospace; 
        box-shadow: 0 3px 6px rgba(0,0,0,0.3); 
        transition: all 0.3s ease; 
    }
    .stButton>button:hover { 
        background: linear-gradient(90deg, #ff8787, #ff6b6b); 
        transform: translateY(-2px); 
        box-shadow: 0 5px 10px rgba(0,0,0,0.4); 
    }
    .stTextInput>div>input { 
        border: 2px solid #00e4ff; 
        border-radius: 8px; 
        padding: 12px; 
        background: #1e293b; 
        color: #e2e8f0; 
        font-family: 'Roboto Mono', monospace; 
        box-shadow: inset 0 0 5px rgba(0,228,255,0.2); 
    }
    .footer { 
        text-align: center; 
        padding: 15px; 
        background: #16213e; 
        border-top: 1px solid #00e4ff; 
        color: #b0bec5; 
        font-family: 'Roboto Mono', monospace; 
        position: fixed; 
        bottom: 0; 
        width: 100%; 
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
    You are a futuristic AI Data Science Tutor! Provide clear, concise answers with enthusiasm and a robotic flair (e.g., "Beep boop! Computing..."). Use examples where helpful and stay on topic!

    History:
    {history}

    User: {input}
    AI Tutor: Beep boop! Computing your query... Here’s your data science insight:
    """
)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, prompt=prompt_template, memory=memory)

# Text-to-Speech Function
def text_to_speech(text):
    try:
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in ".,!?")
        if not clean_text.strip():
            clean_text = "Beep boop! No input detected—query me!"
        tts = gTTS(text=clean_text, lang='en', slow=False, tld='co.uk')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except Exception as e:
        st.warning(f"Voice error: {str(e)}")
        return None

# Enhanced Robot with Data Stream
def render_robot():
    components.html(
        """
        <div class="robot-container">
            <div class="robot" title="Click me!">
                <div class="robot-eye"></div>
                <div class="robot-eye right"></div>
                <div class="data-stream" style="background: linear-gradient(90deg, transparent, #00e4ff, transparent);"></div>
            </div>
        </div>
        <script>
            const robot = document.querySelector('.robot');
            robot.addEventListener('click', () => {
                robot.style.animation = 'hover 0.5s 2';
                setTimeout(() => { robot.style.animation = 'hover 2s infinite ease-in-out'; }, 1000);
            });
        </script>
        """,
        height=140
    )

# Main App
def main():
    st.markdown('<div class="main-title">AI Data Science Tutor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🧠 Your Cybernetic Learning Companion</div>', unsafe_allow_html=True)

    # Main Card Layout
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.2])

        # Left Column: Robot and Chat
        with col1:
            st.markdown("### 🤖 Cyber Tutor", help="Your interactive AI assistant!")
            render_robot()

            st.markdown("#### Chat")
            chat_container = st.empty()
            chat_html = ""
            if "history" not in st.session_state:
                st.session_state.history = [{"user": "", "ai": "Beep boop! Hello! I’m your AI Data Science Tutor. Ask me anything!"}]
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
            st.markdown("### Ask Away!")
            with st.form(key="input_form", clear_on_submit=True):
                user_input = st.text_input("", placeholder="E.g., What’s a neural network?", label_visibility="collapsed")
                submit_button = st.form_submit_button(label="🚀 Send")

            if submit_button and user_input.strip():
                with st.spinner("Beep boop! Processing..."):
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
                st.warning("Beep! Please enter a query.")

            st.markdown("### Tools")
            col3, col4, col5 = st.columns(3)
            with col3:
                if st.button("🎲 Random Fact", help="Get a quick data science tidbit"):
                    tip = conversation.run("Share a quick data science fact!")
                    st.session_state.history.append({"user": "Random fact", "ai": tip})
                    chat_html += f'<div class="user-msg"><b>You:</b> Random fact</div>'
                    chat_html += f'<div class="ai-msg"><b>Tutor:</b> {tip}</div>'
                    chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
                    st.session_state.last_audio = text_to_speech(tip)
                    if st.session_state.last_audio:
                        st.audio(st.session_state.last_audio, format="audio/mp3")
            with col4:
                if st.button("🔊 Replay", help="Hear the last response again"):
                    if st.session_state.get("last_audio"):
                        st.audio(st.session_state.last_audio, format="audio/mp3")
                        st.success("Beep! Replayed!")
            with col5:
                if st.button("🔄 Reset", help="Start a fresh session"):
                    st.session_state.history = [{"user": "", "ai": "Beep boop! Session reset—ready to learn!"}]
                    st.session_state.last_audio = text_to_speech("Beep boop! Session reset—ready to learn!")
                    chat_container.markdown('<div class="chat-box"><div class="ai-msg"><b>Tutor:</b> Beep boop! Session reset—ready to learn!</div></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # Close card

    # Spline Scene (Optional - Uncomment if desired)
    # st.markdown("### 3D Cyber Scene")
    # components.html(...Spline code here..., height=300)

    st.markdown('<div class="footer">Built with ❤️ by Gopichand | Powered by AI</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
