import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PINECONE_API_KEY

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


INDEX_NAME = "langgraph-rag-index"


def get_retriever():
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Created new Pinecone index: {INDEX_NAME}")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever()


def add_document_to_vectorstore(text_content: str):
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    documents = text_splitter.create_documents([text_content])

    print(f"Splitting document into {len(documents)} chunks for indexing...")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")