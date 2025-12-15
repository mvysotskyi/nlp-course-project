import streamlit as st
from final_assistant import FinalLegalAssistantRAG
import sys
import os

# Add the current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Legal Assistant RAG", page_icon="⚖️")

st.title("🇺🇦 Legal Assistant RAG Demo")
st.markdown("Ask questions about Ukrainian Criminal Code and Supreme Court decisions.")

@st.cache_resource
def get_assistant():
    return FinalLegalAssistantRAG()

try:
    assistant = get_assistant()
except Exception as e:
    st.error(f"Failed to initialize assistant: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your legal question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal documents..."):
            try:
                response = assistant.run_pipeline(prompt)
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
