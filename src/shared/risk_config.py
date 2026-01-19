RISK_WEIGHTS = {
    "velocity": 0.3,       # 30% weight to tx frequency
    "travel": 0.3,         # 30% weight to geo-anomalies
    "high_value": 0.2,     # 20% weight to large amounts
    "d17_limit": 0.2       # 20% weight to e-wallet specific limits (increased for smurfing)
}

CBDC_PILOT_GOVERNORATES = ["Tunis", "Sfax"] # High security zones

# D17-specific thresholds for smurfing detection
D17_SOFT_LIMIT = 1500.0    # Threshold where audit flags intensify
D17_VELOCITY_CAP = 5       # Maximum normal transactions per 5-min window