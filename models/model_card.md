# Model Card: Tunisian Fraud Detection Prototype

## Model Overview
- **Model Name**: Tunisian Fraud Detection XGBoost Classifier
- **Version**: 0.1.0 prototype
- **Purpose**: Real-time fraud detection for Tunisian digital payment transactions
- **Algorithm**: XGBoost Gradient Boosted Trees (Spark-compatible)

## Features
The model uses the following features extracted from transaction data:

1. `v_count` - Velocity count (number of transactions in 5-minute window)
2. `g_dist` - Geographic diversity (distinct governorates in 5-minute window)
3. `avg_amount` - Average transaction amount in 5-minute window
4. `is_smurfing` - Binary flag indicating if amounts are in smurfing range (1400-1500 TND)
5. `high_velocity_flag` - Binary flag indicating high transaction velocity (>5 transactions)

## Training Data
- **Source**: Silver layer aggregated transaction data
- **Features**: Derived from 5-minute sliding windows of transaction data
- **Target Variable**: Human-reviewed labels where available; heuristic labels are limited to bootstrap experiments

## Model Parameters
- `max_depth`: 6 (controls tree complexity)
- `n_estimators`: 100 (number of boosting rounds)
- `learning_rate`: 0.1 (shrinkage applied to each tree)

## Intended Use
- Real-time fraud scoring in streaming pipeline
- Configurable e-wallet monitoring controls
- Structuring-pattern research using configurable amount and velocity ranges

## Limitations
- Performance dependent on feature distribution similarity between training and inference
- May require retraining as fraud patterns evolve
- Designed specifically for Tunisian payment ecosystem characteristics
- No model artifact in this repository should be interpreted as independently validated for production decisions
- Hosted API deployment does not currently execute the Spark training workflow

## Ethical Considerations
- Model should not discriminate based on non-financial attributes
- Regular bias audits recommended
- False positive impact on legitimate users should be monitored

## Version History
- 0.1.0 prototype (June 2026): Explicit prototype scope and separated offline/streaming training workflows
