from config import MatcherConfig

def is_match(similarity: float, config: MatcherConfig) -> bool:
    """Determines if a similarity score meets the configured acceptance threshold."""
    return similarity >= config.threshold