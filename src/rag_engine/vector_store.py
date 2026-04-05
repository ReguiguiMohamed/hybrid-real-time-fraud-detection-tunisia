# src/rag_engine/vector_store.py
import logging
import chromadb
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# CTAF regulatory documents to seed the knowledge base
CTAF_REGULATIONS = [
    {
        "id": "ctaf_circular_2024_03",
        "text": (
            "Circular 2024-03: Mobile wallet transfers exceeding 1500 TND per 24h period "
            "require enhanced diligence. Geographic diversity of >2 governorates within "
            "60 minutes constitutes high-risk impossible travel."
        ),
        "metadata": {"source": "ctaf_circular_2024_03", "category": "mobile_wallet_rules"},
    },
    {
        "id": "ctaf_circular_2024_05",
        "text": (
            "Circular 2024-05: All financial institutions must file Suspicious Activity Reports "
            "(SARs) within 10 business days of detection. Reports must include: transaction ID, "
            "user identification, amount, geographic location, payment method, ML risk score, "
            "and a narrative describing the suspicious pattern. Failure to comply may result in "
            "sanctions under Article 78 of Law 2015-26."
        ),
        "metadata": {"source": "ctaf_circular_2024_05", "category": "sar_filing_requirements"},
    },
    {
        "id": "ctaf_circular_2025_01",
        "text": (
            "Circular 2025-01: Enhanced due diligence for digital payment platforms (D17, Flouci, "
            "Konnect). Transactions above 2000 TND via e-wallet require real-time verification. "
            "Smurfing detection threshold set at cumulative 5000 TND across multiple transactions "
            "within a 24-hour window from the same user. Velocity cap of 5 transactions per "
            "5-minute window triggers automatic review."
        ),
        "metadata": {"source": "ctaf_circular_2025_01", "category": "ewallet_compliance"},
    },
    {
        "id": "bct_note_2025_02",
        "text": (
            "BCT Note 2025-02: Cross-governorate transfers must be flagged when a single user "
            "initiates transactions from more than 2 distinct governorates within 1 hour. "
            "This is classified as impossible travel and must be escalated to the compliance "
            "officer within 4 hours of detection. Applies to all payment methods."
        ),
        "metadata": {"source": "bct_note_2025_02", "category": "impossible_travel"},
    },
    {
        "id": "ctaf_aml_guidelines_2025",
        "text": (
            "CTAF AML Guidelines 2025: Risk-based approach to customer due diligence. "
            "High-risk categories include: politically exposed persons (PEPs), cash-intensive "
            "businesses, non-resident accounts, and transactions involving border governorates "
            "(Medenine, Tataouine, Jendouba, Le Kef). Enhanced monitoring required for "
            "transactions during non-business hours (22:00-06:00) exceeding 500 TND."
        ),
        "metadata": {"source": "ctaf_aml_guidelines_2025", "category": "aml_cdd"},
    },
]


class CTAFVectorStore:
    def __init__(self, persist_directory="./data/vector_db"):
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="ctaf_regulations",
                metadata={"hnsw:space": "cosine"}
            )
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._initialize_knowledge_base()
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.client = None
            self.collection = None
            self.embedding_model = None

    def _initialize_knowledge_base(self):
        """Load CTAF regulations into the vector store"""
        if self.collection is None:
            return

        try:
            existing_ids = set()
            if self.collection.count() > 0:
                existing_data = self.collection.get()
                existing_ids = set(existing_data["ids"]) if existing_data and existing_data.get("ids") else set()

            kb_dir = Path("./data/knowledge_base")
            kb_dir.mkdir(parents=True, exist_ok=True)

            new_docs = [r for r in CTAF_REGULATIONS if r["id"] not in existing_ids]
            if not new_docs:
                return

            self.collection.add(
                documents=[r["text"] for r in new_docs],
                ids=[r["id"] for r in new_docs],
                metadatas=[r["metadata"] for r in new_docs],
            )

            # Save to knowledge_base files for reference
            for reg in new_docs:
                filepath = kb_dir / f"{reg['id']}.txt"
                filepath.write_text(reg["text"], encoding="utf-8")

            logger.info(f"Loaded {len(new_docs)} new CTAF regulation documents into vector store")

        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")

    def query(self, query_text, n_results=2):
        """Query the vector store for relevant regulations"""
        if self.collection is None or self.embedding_model is None:
            logger.warning("Vector store not initialized, returning empty results")
            return {"documents": [[]], "ids": [[]], "distances": [[]]}

        try:
            embeddings = self.embedding_model.encode([query_text]).tolist()
            results = self.collection.query(
                query_embeddings=embeddings,
                n_results=min(n_results, self.collection.count()) if self.collection.count() > 0 else 1
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return {"documents": [[]], "ids": [[]], "distances": [[]]}