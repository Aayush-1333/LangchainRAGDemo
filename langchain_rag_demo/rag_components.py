import logging
import os.path
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger("langchain_rag_demo")


def initialize_llm():
    """Start the LLM OpenAI Client which is connecting to vLLM server"""

    logger.info("Setting up Open AI client")
    llm = ChatOpenAI(
        model="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0.5,
        base_url="http://localhost:8200/v1",
        api_key=os.environ["HF_API_TOKEN"]
    )


def initilaize_vector_store():
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
        loader = PyPDFLoader("langchain_rag_demo/data/1. Intermediate Algebra 2e, Lynn Marecek.pdf")
        docs = []
        for doc in loader.load():
            docs.append(doc)

        # splitting docs into chunks
        logger.debug("Splitting doc into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False
        )
        doc_chunks = text_splitter.split_documents(docs)

        # creating vector store
        logger.debug("Saving faiss index into directory: %s", INDEX_PATH)
        vector_store = FAISS.from_documents(doc_chunks, embedding=EMBEDDING_MODEL)
        vector_store.save_local("faiss_index")
    else:
        logger.info("Loading faiss index")
        vector_store = FAISS.load_local(INDEX_PATH, embeddings=EMBEDDING_MODEL, allow_dangerous_deserialization=True)
