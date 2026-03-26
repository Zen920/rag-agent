from langchain_core.tools import tool
from app.agent.chroma import vector_store

@tool(description="Retrieve context by interrogating the provided vector store with the user query.")
def retrieve_context(query: str):
    results = vector_store.vector_store.similarity_search_by_vector_with_relevance_scores(vector_store.embeddings.embed_query(query), k=5)

    return [document.page_content for document, score in results]