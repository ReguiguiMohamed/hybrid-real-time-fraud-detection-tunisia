RISK_WEIGHTS = {
    "velocity": 0.3,
    "travel": 0.3,
    "high_value": 0.2,
    "d17_limit": 0.2,
}

# Prototype controls. These are configurable monitoring defaults, not claims
# about statutory payment limits.
VELOCITY_REVIEW_THRESHOLD = 3
TRAVEL_REVIEW_THRESHOLD = 1
HIGH_VALUE_REVIEW_THRESHOLD_TND = 15000.0
EWALLET_REVIEW_THRESHOLD_TND = 2000.0
ALERT_SCORE_THRESHOLD = 0.5
PROXY_LABEL_THRESHOLD = 0.8
STRUCTURING_AMOUNT_MIN_TND = 1400.0
STRUCTURING_AMOUNT_MAX_TND = 1500.0

CBDC_PILOT_GOVERNORATES = ["Tunis", "Sfax"]

# Backward-compatible names used by the streaming prototype.
D17_SOFT_LIMIT = STRUCTURING_AMOUNT_MAX_TND
D17_VELOCITY_CAP = 5
