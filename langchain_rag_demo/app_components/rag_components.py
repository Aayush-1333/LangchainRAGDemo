import logging
import os.path
from typing import Literal, TypedDict, List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

import streamlit as st
from langgraph.constants import START
from langgraph.graph import StateGraph


logger = logging.getLogger("langchain_rag_demo")
load_dotenv()

template = """You are a smart mathematician. If you do not know the answer, do not try to
make up the answer, just say you don't know. Try to go through the context given below:

{context}

Question: {question}

Helpful answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)


def initialize_llm() -> ChatOpenAI:
    """Start the LLM OpenAI Client which is connecting to vLLM server"""

    logger.info("Setting up Open AI client")
    llm = ChatOpenAI(
        model="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0.5,
        base_url="http://localhost:8200/v1",
        api_key=os.environ["HF_API_TOKEN"]
    )

    return llm


def initialize_vector_store_retriever() -> VectorStoreRetriever:
    """Load vector store if exists, else create new vector store"""

    INDEX_PATH: Literal[str] = "langchain_rag_demo/faiss_index"

    # instantiate hf embedding model
    logger.info("Setting up embedding model")
    EMBEDDING_MODEL = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    if not os.path.exists(INDEX_PATH):
        logger.warning("%s does not exist", INDEX_PATH)

        # loading PDF docs as Document objects
        logger.debug("Loading PDF docs...")
        loader = DirectoryLoader(
            path="langchain_rag_demo/data",
            loader_cls=PyPDFLoader,
            use_multithreading=True
        )
        docs = []
        for doc in loader.load():
            docs.append(doc)

        # splitting docs into chunks
        logger.debug("Splitting doc into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False
        )
        doc_chunks = text_splitter.split_documents(docs)

        # creating vector store
        logger.debug("Saving faiss index into directory: %s", INDEX_PATH)
        vector_store = FAISS.from_documents(doc_chunks, embedding=EMBEDDING_MODEL)
        vector_store.save_local(INDEX_PATH)
    else:
        logger.info("Loading faiss index")
        vector_store = FAISS.load_local(INDEX_PATH, embeddings=EMBEDDING_MODEL, allow_dangerous_deserialization=True)

    # return vector_store.as_retriever(search_type="similarity_score_threshold",
    #                                  search_kwargs={"k": 5, "lambda_mult": 0, "score_threshold": 0.5})
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25})


chat_model = initialize_llm()
vector_store_retriever = initialize_vector_store_retriever()


class State(TypedDict):
    """Langgraph State class"""
    question: str
    context: List[Document]
    answer: str
    metadata_list: List[Dict[str, Any]]


def retrieve(state: State) -> Dict:
    """Langgraph state for retrieving docs from vector DB."""

    logger.debug("Retrieving docs from vector store based on question: %s", state['question'])
    retrieved_docs = vector_store_retriever.invoke(state['question'])
    doc_sources = [doc.metadata for doc in retrieved_docs]
    return {'context': retrieved_docs, 'metadata_list': doc_sources}


def generate(state: State):
    """langgraph state for generating response based on user question and
    context passed as prompt to the model."""

    docs_content = "\n\n".join(doc.page_content for doc in state['context'])
    messages = custom_rag_prompt.invoke({
        'question': state['question'],
        'context': docs_content
    })
    response = chat_model.invoke(messages)
    logger.debug("Invoking chat model to generate response:\n%s", response.content)
    return {'answer': response.content, 'metadata_list': state.get('metadata_list', [])}


@st.cache_resource
def create_rag_graph():
    logger.info("Building and compiling RAG graph...")
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph
