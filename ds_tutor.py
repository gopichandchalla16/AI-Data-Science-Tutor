import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from gtts import gTTS
import os
import time
import io

# Set Streamlit Page Config
st.set_page_config(page_title="AI Data Science Guru", page_icon="📊", layout="wide")

# Custom CSS for Enhanced Styling with Footer and Unique Texts
st.markdown("""
    <style>
    .main-title { font-size: 3em; color: #2ecc71; text-align: center; font-weight: bold; font-family: 'Arial', sans-serif; margin-bottom: 10px; }
    .subtitle { text-align: center; color: #7f8c8d; font-size: 1.2em; margin-bottom: 10px; }
    .motivational { text-align: center; color: #e74c3c; font-size: 1.1em; font-style: italic; transform: rotate(-2deg); margin-bottom: 20px; }
    .chat-box { border: 2px solid #3498db; border-radius: 12px; padding: 20px; background: linear-gradient(135deg, #f5f7fa, #c3cfe2); box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px; min-height: 400px; overflow-y: auto; }
    .user-msg { background-color: #ecf0f1; padding: 12px; border-radius: 8px; margin: 8px 0; font-size: 1.1em; }
    .ai-msg { background-color: #27ae60; color: white; padding: 15px; border-radius: 10px; margin: 8px 0; font-size: 1.3em; font-weight: bold; animation: fadeIn 0.5s ease-in; }
    .did-you-know { position: fixed; right: 20px; top: 150px; width: 200px; background-color: #f1c40f; padding: 15px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-size: 0.9em; color: #2c3e50; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .stButton>button { background-color: #e74c3c; color: white; font-size: 16px; padding: 12px 25px; border-radius: 8px; margin: 5px; }
    .stButton>button:hover { background-color: #c0392b; }
    .stTextInput>div>input { border: 2px solid #3498db; border-radius: 8px; padding: 10px; font-size: 1.1em; }
    .interactive-btn { background-color: #3498db; }
    .interactive-btn:hover { background-color: #2980b9; }
    .footer { text-align: center; padding: 20px; background-color: #ecf0f1; border-top: 2px solid #3498db; font-size: 1em; color: #2c3e50; position: fixed; bottom: 0; width: 100%; box-shadow: 0 -2px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Load Gemini API key
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("⚠️ Oops! Please set the GEMINI_API_KEY in Streamlit secrets to continue!")
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

# Function to Convert Text to Speech and Return Audio Bytes with Male-like Voice
def text_to_speech(text):
    try:
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in ".,!?")
        if not clean_text.strip():
            clean_text = "I’ve got nothing to say—let’s try something else!"
        # Use 'co.uk' TLD for a British male-like voice, which is deeper and clearer
        tts = gTTS(text=clean_text, lang='en', slow=False, tld='co.uk')
        audio_file = "response.mp3"
        tts.save(audio_file)
        if os.name == 'nt':
            os.system(f"start {audio_file}")
        elif os.name == 'posix':
            os.system(f"afplay {audio_file}" if 'darwin' in os.uname().sysname.lower() else f"mpg123 {audio_file}")
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        return audio_bytes
    except Exception as e:
        st.warning(f"Voice-over hiccup: {str(e)}—sticking to text for now!")
        return None

# Function to Replay Audio
def replay_audio(audio_bytes):
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")

# Main Streamlit UI
def main():
    st.markdown('<div class="main-title">AI Data Science Guru</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your Chatty Data Science Pal—Ready to Talk and Teach! 📈🎙️</div>', unsafe_allow_html=True)
    st.markdown('<div class="motivational">Unleash Your Data Science Superpowers with Me!</div>', unsafe_allow_html=True)

    # Did You Know Box
    st.markdown("""
        <div class="did-you-know">
            <b>Did You Know?</b><br>
            The term "Data Science" was coined in 2001 by William S. Cleveland—pretty cool, right?
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        st.session_state.last_audio = None
        ai_greeting = "Hey, hey! I’m your AI Data Science Guru! Super pumped to chat with you about data science. What’s sparking your curiosity today?"
        st.session_state.conversation_history.append({"user": "", "ai": ai_greeting})
        st.session_state.last_audio = text_to_speech(ai_greeting)

    # Chat Interface
    st.markdown("### Let’s Chat!")
    with st.container():
        chat_container = st.empty()
        chat_html = ""
        for exchange in st.session_state.conversation_history:
            if exchange["user"]:
                chat_html += f'<div class="user-msg"><b>You:</b> {exchange["user"]}</div>'
            if exchange["ai"]:
                chat_html += f'<div class="ai-msg"><b>AI Buddy:</b> {exchange["ai"]}</div>'
        chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
        chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)

    # Input Section
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.form(key="input_form", clear_on_submit=True):
            user_input = st.text_input("Ask me anything about data science!", placeholder="E.g., What’s a decision tree?")
            submit_button = st.form_submit_button(label="Send It!")
    with col2:
        st.markdown("#### Voice Input (Upload WAV)")
        audio_file = st.file_uploader("Upload an audio question", type=["wav", "mp3"], key="audio_input")
        if audio_file:
            st.warning("Voice input simulation: Please type your question for now as audio processing isn’t fully supported!")

    # Handle User Input
    if submit_button and user_input.strip():
        with st.spinner("Cooking up an awesome answer... 🤓"):
            ai_response = generate_response(user_input)
            st.session_state.conversation_history.append({"user": user_input, "ai": ai_response})
            chat_html += f'<div class="user-msg"><b>You:</b> {user_input}</div>'
            chat_html += f'<div class="ai-msg"><b>AI Buddy:</b> {ai_response}</div>'
            chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
            chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
            st.session_state.last_audio = text_to_speech(ai_response)
            if st.session_state.last_audio:
                st.audio(st.session_state.last_audio, format="audio/mp3")
            time.sleep(0.5)

    elif submit_button and not user_input.strip():
        st.warning("Hey, don’t leave me hanging! Give me something to work with!")

    # Interactive Features
    st.markdown("### Fun Stuff!")
    col3, col4, col5 = st.columns(3)
    with col3:
        if st.button("Gimme a Random Tip!", key="tip_btn", help="Get a fun data science tip!"):
            with st.spinner("Fetching a cool tip..."):
                random_tip = generate_response("Give me a quick random data science tip!")
                st.session_state.conversation_history.append({"user": "Random tip request", "ai": random_tip})
                chat_html += f'<div class="user-msg"><b>You:</b> Random tip request</div>'
                chat_html += f'<div class="ai-msg"><b>AI Buddy:</b> {random_tip}</div>'
                chat_html += '<script>document.querySelector(".chat-box").scrollTop = document.querySelector(".chat-box").scrollHeight;</script>'
                chat_container.markdown(f'<div class="chat-box">{chat_html}</div>', unsafe_allow_html=True)
                st.session_state.last_audio = text_to_speech(random_tip)
                if st.session_state.last_audio:
                    st.audio(st.session_state.last_audio, format="audio/mp3")

    with col4:
        if st.button("Repeat Last Response", key="repeat_btn", help="Hear the last AI response again!"):
            if st.session_state.last_audio:
                replay_audio(st.session_state.last_audio)
                st.success("Replaying the last thing I said!")
            else:
                st.warning("Nothing to repeat yet—ask me something first!")

    with col5:
        if st.button("Clear Chat", key="clear_btn", help="Start fresh!"):
            st.session_state.conversation_history = []
            st.session_state.last_audio = None
            chat_container.markdown('<div class="chat-box"></div>', unsafe_allow_html=True)
            st.success("Chat cleared! Let’s start over!")

    # Footer Section
    st.markdown("""
        <div class="footer">
           Built with ❤️ by Gopichand | AI That Guides, Teaches & Debugs
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
