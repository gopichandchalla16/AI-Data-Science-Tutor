import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from gtts import gTTS
import os

# Set Streamlit Page Config
st.set_page_config(page_title="AI Data Science Tutor", page_icon="🚀", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.8em; color: #2ecc71; text-align: center; font-weight: bold; font-family: 'Arial', sans-serif; }
    .answer-box { border: 2px solid #2ecc71; border-radius: 12px; padding: 15px; background: linear-gradient(135deg, #e3f2fd, #b8e994); box-shadow: 0 4px 8px rgba(0,0,0,0.15); margin-top: 20px; }
    .stButton>button { background-color: #3498db; color: white; font-size: 16px; padding: 10px 20px; }
    .stButton>button:hover { background-color: #2980b9; }
    </style>
""", unsafe_allow_html=True)

# Load Gemini API key
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("⚠️ Please set the GEMINI_API_KEY in Streamlit secrets to proceed!")
    st.stop()

# Custom LLM Wrapper for Gemini API
class GeminiLLM(LLM):
    def _call(self, prompt: str, stop=None):
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            return response.text if response and hasattr(response, "text") else "⚠️ No response from Gemini API."
        except Exception as e:
            return f"⚠️ Error generating response: {str(e)}"

    def predict(self, prompt: str):
        return self._call(prompt)

    @property
    def _llm_type(self):
        return "Gemini"

# Initialize Gemini LLM
llm = GeminiLLM()

# Define the Prompt Template
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    You are an expert AI Data Science Tutor. Your task is to resolve data science doubts of the user. 
    Respond in a friendly, conversational tone as if you are talking to the user directly.
    Keep your responses concise, accurate, and relevant to data science.

    Conversation History:
    {history}

    User: {input}
    AI Tutor:
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
        return f"⚠️ Error generating response: {str(e)}"

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("start response.mp3")  # Play the audio file (works on Windows)
    # For macOS/Linux, use: os.system("afplay response.mp3")

# Main Streamlit UI
def main():
    st.markdown('<div class="main-title">AI Data Science Tutor</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Your AI-powered Data Science tutor! 🚀</p>", unsafe_allow_html=True)

    # Initialize session state for conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        # AI starts the conversation
        ai_greeting = "Hello! I'm your AI Data Science Tutor. What topic would you like to learn about today?"
        st.session_state.conversation_history.append({"user": "", "ai": ai_greeting})
        # Speak the greeting
        text_to_speech(ai_greeting)

    # Display Conversation History
    st.markdown("### Conversation History")
    for exchange in st.session_state.conversation_history:
        if exchange["user"]:  # Display user input if it exists
            st.markdown(f"**You:** {exchange['user']}")
        if exchange["ai"]:  # Display AI response
            st.markdown(f"**AI Tutor:** {exchange['ai']}")
        st.markdown("---")

    # Input Box
    user_input = st.text_input("Your Question:", placeholder="E.g. What is logistic regression?")

    # Button to Generate Response
    if st.button("Ask"):
        if user_input.strip():
            with st.spinner("Thinking... 💭"):
                # Generate AI response
                ai_response = generate_response(user_input)
                # Update conversation history
                st.session_state.conversation_history.append({"user": user_input, "ai": ai_response})
                # Display AI response
                st.markdown(f'<div class="answer-box"><b>AI Tutor Says:</b><br>{ai_response}</div>', unsafe_allow_html=True)
                # Convert AI response to speech
                try:
                    text_to_speech(ai_response)
                except Exception as e:
                    st.warning("Text-to-speech is not supported in this environment.")
        else:
            st.warning("⚠️ Please enter a question.")

if __name__ == "__main__":
    main()
