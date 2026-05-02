# src/rag_engine/vector_store.py
import logging
import chromadb
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

logger = logging.getLogger(__name__)


# Internal compliance-control documents used to seed the local knowledge base.
# These are not official circular texts. Official regulatory source material should
# be loaded from a verified document mirror before production use.
CTAF_REGULATIONS = [
    {
        "id": "internal_mobile_wallet_controls",
        "text": (
            "Internal control: mobile wallet transfers exceeding configured provider limits "
            "require enhanced diligence. Geographic diversity of more than two governorates "
            "within 60 minutes constitutes high-risk impossible travel."
        ),
        "metadata": {"source": "internal_mobile_wallet_controls", "category": "mobile_wallet_rules"},
    },
    {
        "id": "internal_sar_filing_controls",
        "text": (
            "Internal control: financial institutions must file Suspicious Activity Reports "
            "(SARs) within 10 business days of detection. Reports must include: transaction ID, "
            "user identification, amount, geographic location, payment method, ML risk score, "
            "and a narrative describing the suspicious pattern. Failure to comply may result in "
            "a fine up to TND 50,000 or license revocation."
        ),
        "metadata": {"source": "internal_sar_filing_controls", "category": "sar_filing_requirements"},
    },
    {
        "id": "internal_ewallet_velocity_controls",
        "text": (
            "Internal control: enhanced due diligence applies to digital payment platforms "
            "(D17, Flouci, Konnect). The repealed TND 5,000 cash-payment cap must not be used "
            "as a structuring rule. Smurfing review is based on configurable velocity and "
            "aggregate-amount rules across multiple transactions from the same account."
        ),
        "metadata": {"source": "internal_ewallet_velocity_controls", "category": "ewallet_compliance"},
    },
    {
        "id": "internal_impossible_travel_controls",
        "text": (
            "Internal control: cross-governorate transfers must be flagged when a single user "
            "initiates transactions from more than 2 distinct governorates within 1 hour. "
            "This is classified as impossible travel and must be escalated to the compliance "
            "officer within 4 hours of detection. Applies to all payment methods."
        ),
        "metadata": {"source": "internal_impossible_travel_controls", "category": "impossible_travel"},
    },
    {
        "id": "internal_aml_cdd_controls",
        "text": (
            "Internal control: risk-based approach to customer due diligence. "
            "High-risk categories include: politically exposed persons (PEPs), cash-intensive "
            "businesses, non-resident accounts, and transactions involving border governorates "
            "(Medenine, Tataouine, Jendouba, Le Kef). Enhanced monitoring required for "
            "transactions during non-business hours (22:00-06:00) exceeding 500 TND."
        ),
        "metadata": {"source": "internal_aml_cdd_controls", "category": "aml_cdd"},
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
