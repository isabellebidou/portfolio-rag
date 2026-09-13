import os
import sys

import streamlit as st

# ChromaDB sqlite3 fix
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from app import build_vectorstore  # Imports your existing RAG function!

st.set_page_config(page_title="Isabelle's Personal Minister", page_icon="🤖")
st.title("Isabelle Bidou's Personal Minister 😻")
st.write("Ask me anything about Isabelle's background, skills, and experience!")

# Initialize your LangChain RAG pipeline inside Streamlit's state cache
if "chain" not in st.session_state:
    with st.spinner("Initializing AI Minister..."):
        st.session_state.chain = build_vectorstore()

# Initialize empty chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Isabelle's AI assistant. How can I help you today?"}
    ]

# Display existing chat historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input Box
if user_query := st.chat_input("Ask a question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Generate AI Response
    with st.chat_message("assistant"), st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"input": user_query})
            answer_data = response['answer']
            
            # Extract content string safely from LangChain message object
            clean_answer = answer_data.content if hasattr(answer_data, 'content') else str(answer_data)
            
            st.write(clean_answer)
            st.session_state.messages.append({"role": "assistant", "content": clean_answer})
