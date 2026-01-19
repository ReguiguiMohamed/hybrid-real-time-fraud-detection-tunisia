# src/rag_engine/vector_store.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

class CTAFVectorStore:
    def __init__(self, persist_directory="./data/vector_db"):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create collection for CTAF regulations
        self.collection = self.client.get_or_create_collection(
            name="ctaf_regulations",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load knowledge base if not already populated
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Load CTAF regulations into the vector store"""
        # Check if collection is empty
        if self.collection.count() == 0:
            # Create knowledge base directory if it doesn't exist
            os.makedirs("./data/knowledge_base", exist_ok=True)
            
            # Create the regulation document
            regulation_text = """Circular 2024-03: Mobile wallet transfers exceeding 1500 TND per 24h period require enhanced diligence. Geographic diversity of >2 governorates within 60 minutes constitutes high-risk impossible travel."""
            
            # Add to vector store
            self.collection.add(
                documents=[regulation_text],
                ids=["ctaf_circular_2024_03"],
                metadatas=[{"source": "ctaf_circular_2024_03", "category": "mobile_wallet_rules"}]
            )
            
            # Also save to file for reference
            with open("./data/knowledge_base/ctaf_circular_2024_03.txt", "w", encoding="utf-8") as f:
                f.write(regulation_text)
    
    def query(self, query_text, n_results=2):
        """Query the vector store for relevant regulations"""
        embeddings = self.embedding_model.encode([query_text]).tolist()
        
        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=n_results
        )
        
        return results