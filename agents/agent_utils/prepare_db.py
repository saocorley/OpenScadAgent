import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def prepare_vector_database(file_paths: List[str]):
    """Loads the txt file and returns a vector database. This is done in memory so is done in each loading"""
    text = ""
    for file_path in file_paths:
        with open(file_path, "r") as file:
            text += "\n\n"
            text += "--------------------------------\n\n"
            text += file_path + "\n\n"
            text += file.read()

    # Create text splitter optimized for Spanish text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    # Split documents into chunks
    chunks = text_splitter.split_text(text)

    # Create vector store and BM25 retriever
    vector_store = InMemoryVectorStore.from_texts(chunks, OpenAIEmbeddings())
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 3  # Number of documents to retrieve

    return vector_store, bm25_retriever


def find_relevant_context(vector_store, bm25_retriever, query):
    """Finds the most relevant context using hybrid search (vector + BM25)"""
    # Get results from both retrievers
    vector_docs = vector_store.similarity_search(query, k=3)
    bm25_docs = bm25_retriever.invoke(query)
    # bm25_scores = bm25_retriever.get_scores()  # Get BM25 scores

    # Print scores for debugging
    logging.info("\n **Retrieval Results:**")
    logging.info("\nVector Search Results:")
    for doc in vector_docs:
        print(f"Preview: {doc.page_content}")

    logging.info("\nBM25 Search Results:")
    logging.info("--------------------------------")
    for doc in bm25_docs:
        print(f"Preview: {doc.page_content}")

    # Return all documents
    return vector_docs + bm25_docs
