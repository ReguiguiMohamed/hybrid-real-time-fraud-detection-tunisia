# Future Work — `hybrid-real-time-fraud-detection-tunisia`

> **Status:** Planning document. No code changes yet.  
> **Ground truth date:** May 1, 2026  
> **Intended reader:** Any AI agent or developer picking up a work item cold.  
> All regulatory facts are sourced from verifiable 2025–2026 publications (AMEF Consulting, KPMG, Sovos, African Manager, VATupdate, Zawya, etc.). No synthetic assumptions.

---

## May 2026 Verified Context (Read Before Working on Any Item)

| Domain | Verified Fact | Source Signal |
|---|---|---|
| FATF status | Tunisia exited the FATF grey list and the EU high-risk list. Now ranked **4th lowest** AML risk in Africa (behind Botswana, Seychelles, Mauritius) on the 2025 Basel AML Index. | AMEF Consulting Jan 2026 retrospective |
| STRs to CTAF | 804 STRs filed in 2023 (+52 % vs 529 in 2022). Banks + La Poste account for **93.28 %** of all STRs. 10-business-day filing deadline enforced; non-compliance: up to TND 50,000 fine or license revocation. | African Manager / CTAF |
| Cheque reform | Law n°2024-41 (Aug 2, 2024). **TuniChèque** went live Feb 2, 2025: real-time QR-code provision verification, 8-day clearing window, prison + 20 % penalty for unprovided cheques > TND 5,000. | BCT / La Presse de Tunisie |
| E-invoicing | **Finance Law 2026** (enacted Dec 12, 2025): mandatory e-invoicing expanded to **all** B2B VAT-service transactions from Jan 1, 2026. Clearance model via **TTN (El Fatoora)** platform in real-time. Penalty: TND 100–500/invoice, max TND 50,000/year. | Sovos / KPMG / VATupdate |
| Foreign currency accounts | Finance Law 2026 amendment (voted Dec 2, 2025): Tunisian residents may now open FCY bank accounts — BCT implementation circulars still being drafted as of May 2026. | Zawya / LaunchBase Africa |
| Cash cap | The TND 5,000 cash-payment cap was **repealed** by Finance Law 2026. No hard ceiling currently exists. Smurfing risk vector has shifted but not disappeared. | Finance Law 2026 |
| Digital payments volume | ~TND 27.9 billion (~$8.8 B) in electronic transactions in 2024. Mobile e-wallet growth present but uneven: rural/informal sector largely unserved. Cash remains dominant outside urban centres. | BCT statistics 2024 |
| ML architecture consensus 2025–26 | **GNN + XGBoost hybrid** is the production standard: GNN as a feature factory producing node embeddings → fed into XGBoost classifier. Pure XGBoost on tabular features is no longer considered state-of-the-art for relational fraud. | NVIDIA, Thoughtworks, SAGEPUB 2025 |
| Inference latency floor | ONNX Runtime on XGBoost: ~**200 µs** P50. NVIDIA FIL backend: > 400 K inferences/sec, P99 < 2 ms. Spark RTM (Real-Time Mode, Databricks 2025): < 200 ms for feature engineering pipelines. | NVIDIA / Databricks |
| Streaming engine | Flink remains preferred for **event-by-event** stateful fraud. Spark RTM closed the gap for feature aggregation workloads. A 2025 comparative study showed Spark averaging 0.8 s latency vs Flink 1.7 s at 10 TPS, though Flink scales better at high event cardinality. | ScienceDirect 2025 / Databricks |
| LLM hallucination in finance | LLMs hallucinate on **17–41 %** of finance-domain queries even with RAG. RAG alone is insufficient — regulators require tamper-proof audit logs capturing every retrieval step, prompt, and output modification. | MIT thesis 2025 / Stanford Legal RAG 2025 |
| Deepfake fraud vector | **Deepfake-as-a-Service** platforms emerged as a commoditized fraud vector in 2025–2026, targeting biometric onboarding, KYC selfie liveness, and voice verification. | Biometric Update Jan 2026 |
| Federated learning adoption | SWIFT piloted federated fraud model training with Google Cloud and 12 global banks in 2025 — multi-party model improvement without raw data sharing. | Feedzai / industry reports |
| Behavioral biometrics | Production-grade: typing cadence, mouse dynamics, touchscreen interaction patterns, device orientation. Now table-stakes in tier-1 fintech fraud stacks. Market growing 14.9 % CAGR to $17.8 B by 2030. | PCBB / Feedzai 2025 |

---

## Priority Tiers

Items are ordered **P0 → P4**: P0 blocks compliance/correctness, P4 is research-grade. Within a tier, items are independent unless noted.

---

## P0 — Regulatory Correctness (Blocking — Fix Before Any Demo or Publication)

### P0-1 · Retire the TND 5,000 cash cap rule

**Context:** The Finance Law 2026 repealed the hard TND 5,000 cash-payment ceiling. The current rule engine (e.g., `rules/aml_rules.py` or similar) contains hard-coded thresholds referencing this cap as a structuring trigger.  
**What to do:**  
- Remove or archive any rule that fires solely on the TND 5,000 threshold as a cash-structuring signal.  
- Replace with **velocity-based smurfing detection**: flag sequences where the same payer routes multiple sub-threshold payments (e.g., < TND 3,000 each) to the same beneficiary within a rolling 72-hour window, regardless of absolute amount.  
- Document the legal change in the rule's header comment with a reference to Finance Law 2026, Article X (look up the exact article when implementing).  
- Update any README/architecture docs that mention the TND 5,000 cap.

### P0-2 · Align SAR filing deadline and CTAF penalty language

**Context:** CTAF requires STR filing within **10 business days** (not calendar days). Non-compliance: fine up to TND 50,000 or license revocation. Any prompt template, SAR validator, or compliance README referencing "10 days" without specifying business days must be corrected.  
**What to do:**  
- Audit all SAR generation prompt templates, Pydantic validation schemas, and compliance documentation for the exact phrasing.  
- Standardise to "10 business days (jours ouvrables)" in all user-facing text and in the CTAF submission deadline logic.  
- Add a business-day calendar utility (accounting for Tunisian public holidays) if the system calculates deadlines programmatically.  
- Add the TND 50,000 / license-revocation penalty to any compliance dashboard KPI that tracks overdue SARs.

### P0-3 · Model TuniChèque as an active payment rail

**Context:** TuniChèque went live Feb 2, 2025. It is now the **only** legal pathway for cheques in Tunisia. Cheques outside TuniChèque are invalid. The platform introduces new transaction metadata (QR provision lock, 8-day clearing window, API-connected banks) that creates both new fraud signals and eliminates old ones.  
**What to do:**  
- Add a `TUNICHEQUE` payment channel enum to the transaction schema.  
- Add a rule: flag any cheque transaction that **lacks** a TuniChèque QR verification token (indicative of counterfeit or pre-reform cheque fraud).  
- Add a rule: flag provision-reservation abuse — a legitimate provision lock followed by rapid fund depletion before the 8-day clearing window closes.  
- Remove or deprecate rules designed for the pre-2025 "chèque sans provision" pattern that TuniChèque structurally prevents.

### P0-4 · Add e-invoicing (TTN/El Fatoora) as a new transaction surface

**Context:** As of Jan 1, 2026, **all** B2B service transactions in Tunisia must clear through the TTN El Fatoora platform in real-time before invoice issuance. This creates a new fraud surface: invoice fabrication, VAT carousel fraud, and identity impersonation at the TTN clearance step.  
**What to do:**  
- Add `TTN_EINVOICE` as a transaction type in the data schema.  
- Implement a rule: flag B2B service transactions above TND 1,000 that arrive without a corresponding TTN clearance token.  
- Implement a rule: flag duplicate invoice IDs within a 24-hour window (replay/duplicate submission fraud).  
- Flag merchant accounts with a sudden spike in TTN-cleared invoices within the first 30 days of registration (synthetic merchant fraud).  
- Penalty amounts (TND 100–500/invoice, max TND 50,000/year) should appear in the compliance risk-scoring commentary.

### P0-5 · Add foreign currency account (FCY) monitoring rules

**Context:** Finance Law 2026 allows Tunisian residents to open FCY accounts for the first time. BCT implementation circulars are still being drafted. This creates an immediate layering risk: TND → FCY → offshore.  
**What to do:**  
- Add `FCY_ACCOUNT` as a new account type.  
- Flag FCY accounts opened within 30 days of the BCT circular publication date that immediately receive large TND transfers.  
- Flag round-number TND → FCY conversions (classic layering pattern).  
- Flag FCY accounts receiving funds from multiple TND payers within 7 days (smurfing into FCY).  
- Note in rule documentation that BCT implementation circulars may impose caps — rules must be reconfigurable once circulars are published.

---

## P1 — Model Intelligence Layer (High Priority)

### P1-1 · Replace standalone XGBoost with a GNN → XGBoost hybrid pipeline

**Context:** The 2025–2026 production consensus (NVIDIA, Thoughtworks, multiple peer-reviewed papers) is to use a **Graph Neural Network as a feature factory**: learn node embeddings over a transaction graph (accounts, devices, merchants, IPs as nodes; transactions as edges), then feed those embeddings into XGBoost for the final binary classification. This approach improves detection of ring fraud, smurfing networks, and shared-device abuse without abandoning XGBoost's inference-time speed and interpretability.  
**What to do:**  
- Build a transaction graph: nodes = {account, merchant, device\_fingerprint, IP}; edges = transactions with timestamp and amount attributes.  
- Train a GNN (PyTorch Geometric `GraphSAGE` or `GATv2`) on the graph to produce per-node embeddings. Use inductive sampling so it generalises to new nodes at inference time.  
- At inference: retrieve the embedding for the source account, destination merchant, and device node; concatenate with existing tabular features; pass to XGBoost.  
- Benchmark F1, precision@K, and P95 latency against the current standalone XGBoost baseline. Document delta.  
- The GNN can be retrained weekly (offline); XGBoost retrains daily on the enriched feature set.  
- **Do not** replace XGBoost entirely — the hybrid is the validated approach.

### P1-2 · Add SHAP explainability to every alert

**Context:** BCT/CTAF regulators increasingly require explainable decisions. Compliance officers reviewing alerts need to know which specific features drove a score. LLM-generated SARs that reference model output without grounding in SHAP values are a hallucination amplifier.  
**What to do:**  
- Add `shap.TreeExplainer` to the scoring path. Compute SHAP values per prediction.  
- Store top-5 SHAP feature contributions (feature name, value, SHAP impact) alongside every alert in the alert schema.  
- Feed these top-5 contributions as structured grounding context into the SAR RAG prompt — do not ask the LLM to infer reasons from raw scores alone.  
- Expose SHAP waterfall charts in the Streamlit dashboard per alert.  
- Gate SAR auto-submission on a minimum SHAP confidence threshold (configurable per SAR type).

### P1-3 · Implement behavioral sequence features using a lightweight Transformer encoder

**Context:** XGBoost on tabular snapshots cannot model a user's temporal transaction sequence. A lightweight Transformer (e.g., a 2-layer encoder on the last 50 transactions per user) can encode sequence-level patterns (unusual time-of-day, atypical merchant category sequence, velocity ramp-up) as a fixed-length embedding, which then becomes an additional feature for XGBoost.  
**What to do:**  
- Build a user-session encoder: input = sequence of (amount, merchant\_category, channel, hour\_of\_day, day\_of\_week, delta\_t\_since\_last\_tx) for the last 50 transactions; output = 64-dim embedding.  
- Train with a contrastive loss: embeddings for the same user across different sessions should be similar; embeddings for fraudulent vs. legitimate sessions should be dissimilar.  
- At inference: fetch the last 50 transactions for the source account from the feature store; run the encoder; append the 64-dim embedding to the XGBoost feature vector.  
- Target P95 encoder inference < 5 ms (encoder should be ONNX-exported).  
- This replaces or augments the current "velocity last N hours" hand-crafted features with learned temporal representations.

### P1-4 · Add device fingerprinting and behavioral biometrics signals to the feature schema

**Context:** Production fraud stacks in 2025–2026 treat device fingerprint and behavioral biometrics as table-stakes signals. The current pipeline ingests transaction metadata but has no device-level or interaction-level signals.  
**What to do:**  
- Extend the Kafka event schema to accept (optionally): `device_id`, `device_os`, `device_model`, `app_version`, `session_typing_cadence_ms` (median inter-keystroke delay), `session_copy_paste_ratio`, `network_type` (4G/WiFi/VPN), `vpn_detected` (boolean), `emulator_detected` (boolean).  
- Mark all these fields nullable — they will be absent from API-originated transactions; presence/absence itself is a signal.  
- Add rules: flag transactions where `vpn_detected=True` combined with a new device + high amount. Flag `emulator_detected=True` transactions outright for review.  
- Add features: device velocity (how many accounts transacted from this device\_id in the last 7 days), device age (days since first seen).  
- Do **not** build the biometric capture layer — that is a client SDK concern. Only ingest and model the signals.

### P1-5 · Drift detection and automated retraining triggers

**Context:** The current active learning loop retrains on schedule or manual trigger. Production systems need quantitative drift detection so retraining fires when it matters, not on a fixed calendar.  
**What to do:**  
- Implement Population Stability Index (PSI) on the top-10 XGBoost input features. Compute PSI weekly against the training distribution. Threshold: PSI > 0.2 triggers a retraining job.  
- Implement concept drift detection on model output scores: use Page-Hinkley test on the rolling mean predicted fraud probability. A significant shift triggers retraining.  
- Add a new Grafana panel: "Feature Drift (PSI)" per feature over time.  
- On retraining trigger: log the triggering condition (which feature drifted, PSI value) to the audit trail.  
- Champion-challenger: new retrained model runs in shadow mode for 48 hours before promotion. Promotion gate: new model F1 ≥ current model F1 − 0.005.

### P1-6 · Implement isolation forest as a complementary anomaly detector

**Context:** XGBoost is a supervised classifier. It cannot detect novel fraud patterns with zero labelled examples. An unsupervised isolation forest running in parallel provides a "zero-day" signal.  
**What to do:**  
- Train `sklearn.ensemble.IsolationForest` on all transactions (not just labelled ones) using the same tabular features as XGBoost.  
- At inference: compute the anomaly score alongside the XGBoost probability score.  
- Define an alert tier: `HIGH_ANOMALY` = XGBoost score < 0.4 but isolation forest anomaly score < −0.3 (highly anomalous but not yet classified as fraud by the supervised model). Route these to a human review queue separately from standard fraud alerts.  
- Retrain isolation forest monthly on the full transaction history.

---

## P2 — Infrastructure & Performance

### P2-1 · Export XGBoost to ONNX Runtime for inference

**Context:** ONNX Runtime achieves ~200 µs P50 latency for XGBoost inference (vs. native XGBoost predict which is slower and harder to serve consistently). NVIDIA FIL achieves > 400 K inferences/sec at P99 < 2 ms. The current serving path has not been benchmarked against these targets.  
**What to do:**  
- Export the trained XGBoost model to ONNX using `onnxmltools`.  
- Serve via ONNX Runtime (`onnxruntime.InferenceSession`) in the FastAPI scoring endpoint.  
- Add a latency benchmark test that asserts P95 scoring latency < 5 ms under 500 concurrent requests (using `locust` or `k6`).  
- If a GPU is available in the target deployment environment, evaluate NVIDIA FIL (Triton Inference Server with FIL backend) as the serving layer.  
- Update `README` performance claims only after measured benchmark results.

### P2-2 · Evaluate Spark Real-Time Mode (RTM) or Flink for the stateful aggregation layer

**Context:** Databricks released Spark RTM in 2025, claiming < 200 ms end-to-end latency for feature aggregation pipelines without switching to a second engine. A 2025 peer-reviewed study (ScienceDirect) showed Spark achieving 0.8 s vs. Flink 1.7 s at 10 TPS, though Flink scales better at high cardinality. The current architecture uses Spark Structured Streaming micro-batches, which introduces a latency floor of several seconds.  
**What to do:**  
- Benchmark current Spark Structured Streaming P95 end-to-end latency (Kafka ingest → feature computation → score published) on a representative workload.  
- If latency > 1 s P95: prototype Spark RTM with `trigger(availableNow=False, continuousCheckpointIntervalMs=100)`.  
- If latency > 500 ms and sub-second is required: prototype an Apache Flink job for the velocity/aggregation stateful operators only (not full pipeline replacement).  
- Document the chosen path with benchmark numbers. Do not migrate the entire pipeline unless the benchmark justifies it.  
- Target: P95 end-to-end latency < 500 ms for the alert generation path.

### P2-3 · Load test to validate 1M tx/day claim and identify actual bottlenecks

**Context:** The README claims ~1 M transactions/day throughput. This is approximately 11.6 tx/sec average, very modest. But the claim has not been validated under realistic burst conditions (Tunisian e-wallet peak hours, month-end salary payments, Ramadan commercial spikes).  
**What to do:**  
- Write a `k6` or `locust` load test that simulates: 10x burst (116 tx/sec for 10 minutes), sustained 1 M/day flat, and 5x burst with intentional GNN embedding cache misses.  
- Measure: Kafka consumer lag, Spark executor CPU/memory, XGBoost ONNX P95 latency, ChromaDB RAG query time, alert PostgreSQL write throughput.  
- Identify the actual bottleneck tier. Fix it. Document the real sustainable throughput.  
- Update all README throughput claims to match measured results.

### P2-4 · Replace SHA-256-only PII masking with a format-preserving encryption (FPE) scheme

**Context:** The current SHA-256 hashing of account IDs and phone numbers is one-way and irreversible, which breaks use cases like: compliance officer investigation workflow, linking a flagged account to a CRM record, or deduplicating alerts across time. Format-preserving encryption (FPE, e.g., FF3-1 / AES-FFX) allows PII masking that is reversible by authorised parties while remaining opaque to the data pipeline.  
**What to do:**  
- Implement FF3-1 (NIST SP 800-38G) FPE for account IDs, phone numbers, and national ID fields using a key stored in Vault.  
- SHA-256 hashes can remain for k-anonymity bucketing in analytics, but primary record linking must use FPE tokens.  
- Define two roles in Vault: `pipeline-reader` (no decrypt access) and `compliance-officer` (decrypt access with audit log).  
- Ensure the FPE key rotation procedure is documented.

### P2-5 · Add a real-time feature store for online serving

**Context:** The GNN embeddings (P1-1) and behavioral sequence embeddings (P1-3) must be precomputed and available at sub-millisecond lookup times during scoring. The current architecture has no dedicated online feature store.  
**What to do:**  
- Deploy Redis (or Apache Feast with a Redis online store) as the online feature store.  
- Precompute and store per-account: GNN node embedding (updated every 15 minutes), last-50-tx sequence embedding (updated on every new transaction), velocity counters (updated in the Spark/Flink layer).  
- The FastAPI scoring endpoint should do: (1) fetch precomputed features from Redis in < 1 ms, (2) call ONNX Runtime in < 1 ms, (3) return score.  
- TTL on feature entries: 7 days (accounts not seen in 7 days fall back to a cold-start prior).

### P2-6 · Pin and audit all dependencies for CVE exposure

**Context:** The project uses Docker + Vault but the Python dependency set (Kafka client, Spark, XGBoost, Ollama, ChromaDB, FastAPI, Streamlit) has not been audited against known CVEs as of May 2026. Several high-severity CVEs were published in 2024–2025 for `transformers`, `langchain`, and `chromadb`.  
**What to do:**  
- Run `pip-audit` (or `safety check`) against the pinned `requirements.txt`. Output a CVE report.  
- Pin all transitive dependencies to a known-good hash in `requirements.txt` (use `pip-compile --generate-hashes`).  
- Add a GitHub Actions / CI step that blocks merges if `pip-audit` reports a CRITICAL or HIGH CVE.  
- For Docker base images: switch from `latest` tags to digest-pinned references.

---

## P3 — Compliance Depth & SAR Quality

### P3-1 · Harden the RAG-based SAR generator against hallucination

**Context:** LLMs hallucinate on 17–41 % of finance-domain queries even with RAG (MIT 2025 thesis; Stanford Legal RAG 2025 paper). In a CTAF-submitted SAR, a hallucinated account number, date, or transaction amount is a regulatory liability. RAG alone is not a sufficient mitigation.  
**What to do:**  
- **Structured grounding first:** Before any LLM call, assemble a structured JSON object containing all factual claims the SAR must make (account IDs, amounts, timestamps, SHAP top-5 features, rule IDs that fired). Pass this JSON as the sole source of facts in the prompt. Instruct the model: "Use only the values in the provided JSON. Do not invent or infer values not present."  
- **Post-generation fact-checking:** Parse the generated SAR text and verify every numeric and identifier claim against the source JSON using regex + fuzzy match. If any discrepancy is found, reject the generation and retry (max 2 retries), then fall through to the deterministic template.  
- **Deterministic fallback:** The deterministic SAR template must always produce a valid, submittable SAR without any LLM involvement. The LLM path is for narrative enrichment only, not for fact generation.  
- **Audit log:** Log every LLM call with: model version, prompt hash, retrieved chunk hashes, raw output, fact-check result, final SAR hash. Store immutably (append-only log).  
- **Human-in-loop gate:** No SAR is auto-submitted to CTAF. A compliance officer must approve each SAR. The dashboard must show the LLM confidence score and any fact-check warnings prominently before the approve button is enabled.

### P3-2 · Add perpetual KYC (pKYC) event triggers

**Context:** BCT Circular 2025-17 on internal controls and the broader 2025–2026 AML trend explicitly move toward **perpetual KYC**: continuous re-verification triggered by risk events, not just at onboarding. Banks are expected to re-screen customers when significant risk signals appear.  
**What to do:**  
- Define a `pKYC_trigger` event type published to a Kafka topic whenever: (a) fraud score > 0.7 for a previously low-risk account, (b) account appears in a newly detected transaction ring (GNN cluster), (c) account opens a new FCY account (P0-5), (d) account's transaction velocity increases > 300 % week-over-week.  
- The `pKYC_trigger` event schema must include: account ID (FPE-masked), trigger reason, timestamp, current risk tier, and the specific signals that triggered it.  
- Downstream consumers of this topic (e.g., a CRM or KYC platform) are out of scope — but the event must be published correctly so any consumer can act.  
- Add a Grafana panel: "pKYC triggers per day" broken down by trigger reason.

### P3-3 · Implement sanctions and PEP screening integration point

**Context:** CTAF/BCT require screening against sanctions lists (UN, EU, US OFAC, BCT local list) and Politically Exposed Persons (PEP) databases. This is entirely absent from the current pipeline.  
**What to do:**  
- Add a `sanctions_screen` step in the transaction processing pipeline that checks `sender_account` and `receiver_account` against a configurable sanctions list (initially: a local CSV mirror of UN Consolidated List + a stub for BCT's list).  
- A hit on sanctions screening should **immediately** produce a `SANCTIONS_HIT` alert (highest severity, bypassing normal ML scoring) and freeze the transaction pending compliance review.  
- Add PEP flag as a boolean field in the account enrichment schema. PEP-connected transactions automatically escalate to Enhanced Due Diligence (EDD) tier.  
- Use a pluggable interface so a real commercial sanctions feed (e.g., Dow Jones, ComplyAdvantage) can replace the CSV stub without pipeline changes.  
- Screen at transaction time, not batch — this is a blocking check.

### P3-4 · Build a compliance KPI dashboard layer

**Context:** Compliance officers at BCT-regulated institutions need to report against specific KPIs, not just see raw alerts. These KPIs are distinct from engineering observability metrics.  
**What to do:**  
- Add a dedicated Grafana dashboard (or Streamlit tab) labelled "Compliance View" with the following metrics, updated daily:  
  - SARs filed to CTAF in the last 30 days  
  - SARs filed within 10 business days (% on-time)  
  - Overdue SARs (> 10 business days, zero should be the target)  
  - Sanctions hits (last 30 days)  
  - pKYC triggers by reason code  
  - False positive rate on human-reviewed alerts (reviewed and closed as non-fraud / total alerts)  
  - High-risk account count by tier (BCT risk tier taxonomy)  
- This dashboard should be exportable as a PDF for regulatory submission.

### P3-5 · Add explicit audit trail for all rule changes and model promotions

**Context:** Regulators require that any change to fraud detection logic (rule threshold change, model promotion, rule addition/removal) be logged with: who changed it, when, what changed, and why.  
**What to do:**  
- Every rule definition file must be versioned (git tag + semantic version in the file header).  
- On any rule change: write a structured audit event to an append-only log: `{timestamp, actor, rule_id, old_value, new_value, justification, related_regulatory_reference}`.  
- Model promotions (champion-challenger, active learning) must write a similar event: `{timestamp, model_version, previous_version, promotion_trigger, performance_delta, approved_by}`.  
- `approved_by` must be a human identifier — no automated promotion without a human approval record.  
- This log must be tamper-evident (hash-chained entries or write to Vault audit log).

---

## P4 — Ambitious Directions (Research-Grade)

### P4-1 · Federated learning across simulated multi-bank environment

**Context:** SWIFT piloted federated fraud model training with 12 banks and Google Cloud in 2025. In the Tunisian context, BCT could mandate a similar inter-bank consortium. The architecture should be ready for this.  
**What to do:**  
- Implement a **simulated** federated learning setup using `Flower` (flwr) framework with 3 simulated bank "clients" each holding a partition of the transaction data.  
- Train the GNN embedding model (P1-1) in a federated manner: each client trains locally, only gradients (not raw data) are aggregated at the central server using FedAvg.  
- Measure: federated model F1 vs. centrally-trained model F1. Document the accuracy cost of federation.  
- The simulation must use real transaction schema and realistic data partitioning (not random splits) — each "bank" should have its own merchant categories, geographic distribution, and device mix.  
- Output: a reproducible benchmark notebook and an architecture proposal document for BCT submission.

### P4-2 · Deepfake and synthetic identity fraud detection module

**Context:** Deepfake-as-a-Service platforms are commoditized as of 2026 (Biometric Update, Jan 2026). They enable: synthetic face generation bypassing KYC selfie liveness, voice cloning for phone verification, and synthetic identity document generation. The current pipeline has no signal for this vector.  
**What to do:**  
- Add a `DEEPFAKE_RISK` signal field to the account onboarding event schema (boolean + confidence score, populated by an upstream KYC system or liveness check API).  
- Add a rule: if `deepfake_risk_score > 0.6` at onboarding AND the account transacts > TND 500 within 24 hours of creation, auto-escalate to review.  
- Research integration with open-source liveness detection models (e.g., iBeta-certified passive liveness SDKs) as a pluggable component.  
- Add synthetic identity detection: flag accounts where: (a) the national ID number pattern is valid but the ID was never seen in any historical transaction, (b) device fingerprint is shared with > 3 other new accounts registered in the last 7 days, (c) onboarding email domain is < 30 days old.

### P4-3 · Arabic language support in SAR generation and compliance dashboard

**Context:** CTAF accepts SARs in Arabic and French. All Tunisian compliance officers are Arabic/French bilingual. The current pipeline generates SARs in English or French only. Arabic-language SAR templates would meaningfully differentiate this project for Tunisian institutional users.  
**What to do:**  
- Add an Arabic SAR template (deterministic, for the P3-1 fallback path). Use a professional translator or verified Tunisian legal Arabic terminology — do not use machine-translated templates without human review.  
- For the LLM-generated narrative path: evaluate `Jais-13B` (Arabic-English LLM from G42/MBZUAI) or `AraGPT2` via Ollama as a local Arabic generation model.  
- Benchmark Arabic LLM SAR quality against the French LLM path using BLEU and factual accuracy metrics (using the P3-1 fact-check framework).  
- Add a language toggle (AR / FR) to the Streamlit compliance dashboard. RTL layout support required for Arabic mode.

### P4-4 · Transaction graph visualisation for analyst workflow

**Context:** Fraud analysts investigating ring fraud or smurfing networks need to see the transaction graph, not just individual alerts. The current dashboard shows tabular alert lists.  
**What to do:**  
- Add a graph visualisation tab to the Streamlit dashboard using `pyvis` or `streamlit-agraph`.  
- On opening an alert: query Neo4j (or a NetworkX in-memory graph built from recent transactions) for the 2-hop neighbourhood of the flagged account.  
- Render the subgraph with: node colour = risk tier, edge weight = transaction amount, edge label = timestamp + channel.  
- Highlight the specific edges that triggered the GNN anomaly score.  
- Export the graph as a PNG for SAR attachment (CTAF accepts attachments).

### P4-5 · Adversarial robustness evaluation

**Context:** Fraudsters in 2025–2026 use AI to probe and evade fraud detection systems (adversarial ML). The current model has not been evaluated for adversarial robustness.  
**What to do:**  
- Implement a **red team** notebook: use `adversarial-robustness-toolbox` (ART) to generate adversarial transaction examples that minimally perturb features to cross below the XGBoost decision boundary.  
- Measure: minimum perturbation required to evade detection (in TND amount, velocity count, etc.). This gives a concrete "evasion cost" for the adversary.  
- Implement adversarial training: add a fraction of ART-generated adversarial examples to the training set at retraining time.  
- Document which features are most exploited by adversarial perturbation — these are candidates for additional rule-based hardening.

---

## Dependency Map

Some items are prerequisites for others:

```
P0-1, P0-2, P0-3, P0-4, P0-5  →  can be done independently, in parallel
P1-1 (GNN)                     →  required before P2-5 (feature store)
P1-2 (SHAP)                    →  required before P3-1 (SAR hardening, grounding step)
P1-1 + P1-2                    →  required before P4-4 (graph visualisation)
P2-1 (ONNX)                    →  required before P2-5 (feature store serving path)
P3-1 (SAR hardening)           →  required before P3-4 (compliance KPI — on-time SAR metric)
P3-3 (sanctions)               →  required before P3-2 (pKYC — sanctions hit is a pKYC trigger)
P1-5 (drift detection)         →  required before P3-5 (audit trail — model promotions)
```

---

## What This Document Does NOT Include

- Synthetic data generation improvements — the synthetic data layer is a known limitation; real data partnerships with Tunisian fintech operators or BCT are the only path to validation that matters.  
- Generic ML tutorials or introductory refactors — every item above assumes working familiarity with the existing codebase.  
- Cloud migration — the project's data-residency constraint (no PII export, local LLM) is a design requirement, not a gap. Items above respect it.  
- Business metrics fabrication — no throughput, accuracy, or latency claims should be stated in documentation until they are measured on the actual running system.
