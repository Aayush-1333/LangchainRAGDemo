from langchain_rag_demo.logger import setup_logger
from langchain_rag_demo.rag_components import initilaize_vector_store, initialize_llm

from dotenv import load_dotenv
import streamlit as st


if __name__ == '__main__':
    logger = setup_logger("langchain_rag_demo", mode="INFO")
    load_dotenv()

    logger.info("Starting RAG Demo App...")
    st.title("RAG Demo App")

    initialize_llm()
    initilaize_vector_store()
