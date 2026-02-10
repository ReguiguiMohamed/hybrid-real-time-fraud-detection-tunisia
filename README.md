# Amastan - Fraud Shield Guard: Hybrid Streaming & RAG Architecture

A real-time fraud mitigation engine for Tunisian digital payments. Uses **Kafka + Spark Structured Streaming** for millisecond detection and **RAG (Ollama/ChromaDB)** for automated CTAF-compliant reporting.

The topic came into fruition ever since the introduction of incentives on 'cashless' transactions in Tunisia during January-February 2026, a period marked by the highest ever recorded liquidity rate in the country's history amid deepening inflation rates and economic uncertainty.

### January-February 2026 Economic State of Tunisia
- **Liquidity Crisis**: Peak monetary expansion following unprecedented fiscal stimulus measures
- **Inflation Surge**: Double-digit inflation rates destabilizing purchasing power
- **Digital Payment Boom**: Government incentives for cashless transactions to digitize economy
- **Fraud Vulnerability**: Rapid digital adoption creating new attack vectors for financial crime
- **Regulatory Pressure**: CTAF mandates stricter AML/CFT compliance amid economic instability

### Philosophy & Policy Approach
- **Economic Sovereignty**: Financial infrastructure resilience against external shocks
- **Digital Trust**: Secure payment ecosystems as foundation for economic recovery
- **Regulatory Compliance**: Automated adherence to evolving BCT/CTAF requirements
- **Real-time Defense**: Proactive fraud prevention over reactive investigation

### Technical Context
- **Market**: Tunisia mobile payment volume reaching 12% CAGR
- **Compliance**: Automated SAR generation follows the 10-business-day filing mandate (CTAF/BCT)
- **Core Stack**: Kafka 4.1+, Spark 3.5+ (Real-time Mode), Protobuf 5.29+, XGBoost 3.1+
