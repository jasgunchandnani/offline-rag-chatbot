import streamlit as st
from src.rag import answer

st.set_page_config(page_title="Offline RAG Bot")

st.title("📚 Offline RAG Chatbot")

query = st.text_input("Ask a question")

if query:
    with st.spinner("Thinking..."):
        response = answer(query)

    st.markdown("### Answer")
    st.write(response)
