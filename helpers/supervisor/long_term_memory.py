import chromadb
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict, Any

class LongTermMemory:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB for long-term memory storage."""
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        self.vectorstore = Chroma(
            collection_name="agent_memory",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def store_memory(self, content: str, metadata: Dict[str, Any] = None):
        """Store a new memory in the vector database."""
        if metadata is None:
            metadata = {}

        import datetime
        metadata.update({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": metadata.get("type", "general")
        })

        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[metadata]
        )

    def retrieve_memories(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """Retrieve relevant memories based on semantic similarity."""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            memories = []
            for doc in docs:
                memories.append({
                    "role": "memory",
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            # print(f"Retrieved memories: {memories}")
            return memories
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []

    def clear_memories(self):
        """Clear all stored memories (for testing)."""
        try:
            # Delete the collection and recreate
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name="agent_memory",
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

        except Exception as e:
            print(f"Error clearing memories: {e}")


ltm = LongTermMemory()