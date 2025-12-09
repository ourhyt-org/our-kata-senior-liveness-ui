def normalize_score(value: float) -> float:
    if value < 0: return 0.0
    if value > 1: return 1.0
    return round(value, 3)