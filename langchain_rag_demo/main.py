from app_components.logger import setup_logger
from app_components.rag_components import create_rag_graph

import streamlit as st


if __name__ == '__main__':
    logger = setup_logger("langchain_rag_demo", mode="INFO")

    st.title("RAG Demo App")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello, I am Lucy. How can I help you?"
        }]

    rag_graph = create_rag_graph()

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    user_prompt = st.chat_input("Ask something")
    if user_prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": user_prompt
        })
        with st.chat_message("user"):
            st.markdown(user_prompt)

        ai_response = rag_graph.invoke({'question': user_prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_response['answer']
        })
        with st.chat_message("assistant"):
            st.markdown(ai_response['answer'])
