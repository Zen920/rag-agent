import os
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()
DEFAULT_DOCUMENT = os.environ.get("MISC.DEFAULT_DOCUMENT_PATH")
SPLITTER_CHUNK_SIZE = int(os.getenv("SPLITTER.CHUNK_SIZE"))
SPLITTER_OVERLAP = int(os.getenv("SPLITTER.CHUNK_OVERLAP"))
SPLITTER_ADD_START_INDEX = bool(os.getenv("SPLITTER.ADD_START_INDEX"))
STORE_HOST = os.getenv("CHROMADB.HOST")
STORE_PORT = int(os.getenv("CHROMADB.PORT"))
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDINGS.MODEL")

def load_pdf(document : str | None):
    loader =  PyPDFLoader(document or DEFAULT_DOCUMENT)
    return loader.load()


class VectorStoreInterface:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SPLITTER_CHUNK_SIZE, chunk_overlap=SPLITTER_OVERLAP, add_start_index=SPLITTER_ADD_START_INDEX
        )
        self.vector_store = Chroma(collection_name="document_collections", embedding_function=self.embeddings,
                              host=STORE_HOST,
                              port=STORE_PORT,
                                collection_metadata={"hnsw:space": "cosine"}
                                   )
    def add_documents(self, docs):
        """Add documents to the vector store."""
        all_splits = self.text_splitter.split_documents(docs)
        uuids = [str(uuid4()) for _ in range(len(all_splits))]
        self.vector_store.add_documents(documents=all_splits, ids=uuids)

vector_store = VectorStoreInterface()

if __name__ == '__main__':
    vector_store.add_documents(load_pdf(None))