# Tunisian Fraud Guard: Hybrid Streaming & RAG Architecture

A real-time fraud mitigation engine for Tunisian digital payments. Uses **Kafka + Spark Structured Streaming** for millisecond detection and **RAG (Ollama/ChromaDB)** for automated CTAF-compliant reporting.

### 2026 Context
- **Market**: Tunisia mobile payment volume surging at 12% CAGR (Post-2025 e-payment boom).
- **Compliance**: Automated SAR generation follows the 10-business-day filing mandate (CTAF/BCT).
- **Core Stack**: Kafka 4.1+, Spark 3.5+ (Real-time Mode), Protobuf 5.29+, XGBoost 3.1+.