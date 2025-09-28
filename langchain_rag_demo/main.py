from app_components.logger import setup_logger
from app_components.rag_components import create_rag_graph

import streamlit as st


if __name__ == '__main__':
    logger = setup_logger("langchain_rag_demo", mode="DEBUG")

    st.title("RAG Demo App")

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello, I am Lucy. How can I help you?"
        }]

    if "last_ai_response" not in st.session_state:
        st.session_state.last_ai_response = {}

    if "selected_doc_idx" not in st.session_state:
        st.session_state.selected_doc_idx = 0

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

        # Store AI response for dropdown rendering
        st.session_state.last_ai_response = ai_response
        st.session_state.selected_doc_idx = 0


    # Create dropdown labels (e.g., titles or sources)
    last_ai_response = st.session_state.get('last_ai_response', {})
    metadata_list = last_ai_response.get('metadata_list', [])

    dropdown_labels = [meta.get("source", f"Doc {i + 1}") for i, meta in enumerate(metadata_list)]
    selected = st.selectbox(
        "Select a document",
        dropdown_labels,
        index=st.session_state.selected_doc_idx,
        key="doc_dropdown"
    )

    # Optionally show full metadata
    if selected:
        st.session_state.selected_doc_idx = dropdown_labels.index(selected)
        st.json(metadata_list[st.session_state.selected_doc_idx])
