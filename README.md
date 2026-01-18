# Tunisian Fraud Guard: Hybrid Streaming & RAG Architecture

A real-time fraud mitigation engine for Tunisian digital payments. Uses **Kafka + Spark Structured Streaming** for millisecond detection and **RAG (Ollama/ChromaDB)** for automated CTAF-compliant reporting.
The topic came into fruition ever since the introduction of incentives on 'cashless' transactions in Tunisia during january 2026, a period marked by the highest ever recorded liquidity rate in the country's history, deepening inflation rates.
### 2026 Context
- **Market**: Tunisia mobile payment volume reaching 12% CAGR .
- **Compliance**: Automated SAR generation follows the 10-business-day filing mandate (CTAF/BCT).
- **Core Stack**: Kafka 4.1+, Spark 3.5+ (Real-time Mode), Protobuf 5.29+, XGBoost 3.1+.
