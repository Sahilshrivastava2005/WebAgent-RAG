import streamlit as st
import os
import tempfile
from backend.agent import run_agent
from backend.vectorstore import add_document_to_vectorstore

# Page config
st.set_page_config(
    page_title="WebAgent-RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .reportview-container {
        background: #fafafa
    }
    .sidebar .sidebar-content {
        background: #f0f2f6
    }
    .stChatInputContainer {
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stChatMessage {
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f7ff;
    }
    .assistant-message {
        background-color: #ffffff;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for messages and thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bot.png", width=64)
    st.title("WebAgent Settings")
    st.markdown("---")
    
    st.header("Upload Knowledge")
    uploaded_file = st.file_uploader("Upload a text document to Pinecone (.txt)", type=["txt"])
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing and uploading to Pinecone..."):
                try:
                    content = uploaded_file.read().decode("utf-8")
                    add_document_to_vectorstore(content)
                    st.success("Document added successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    
    st.markdown("---")
    st.header("Agent Capabilities")
    web_search_enabled = st.toggle("Enable Web Search (Tavily)", value=True, help="Allow the agent to search the internet if it doesn't know the answer.")
    
    st.markdown("---")
    st.caption("Built with LangGraph & Streamlit")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Main Chat Interface
st.title("🤖 WebAgent-RAG Chat")
st.markdown("Ask questions based on uploaded documents or the web!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Call our langgraph agent
                response = run_agent(
                    query=prompt, 
                    thread_id=st.session_state.thread_id,
                    web_search_enabled=web_search_enabled
                )
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.info("Please ensure all API keys (Groq, Pinecone, Tavily) are set in your .env file.")
