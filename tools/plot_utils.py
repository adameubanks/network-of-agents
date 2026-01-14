import os
from typing import Tuple


# Central short->long topic key mapping for file/dir names to runner mapping keys
SHORT_TO_LONG = {
    "activism": "corporate_activism",
    "cloning": "human_cloning",
    "democracy": "social_media_democracy",
    "economy": "environment_economy",
    "etiquette": "restaurant_etiquette",
    "paper": "toilet_paper",
    "safety": "gun_safety",
    "sandwich": "hot_dog_sandwich",
    "weddings": "child_free_weddings",
    "immigration": "immigration",
}


def to_long_topic_key(short_key: str) -> str:
    return SHORT_TO_LONG.get(short_key, short_key)


def to_display_name(short_or_long_key: str) -> str:
    """Convert topic key to properly capitalized display name."""
    # Special cases for proper formatting
    special_cases = {
        "child_free_weddings": "Child-Free Weddings",
        "hot_dog_sandwich": "Hot Dog Sandwich",
        "social_media_democracy": "Social Media Democracy",
        "restaurant_etiquette": "Restaurant Etiquette",
        "corporate_activism": "Corporate Activism",
        "environment_economy": "Environment Economy",
        "gun_safety": "Gun Safety",
        "human_cloning": "Human Cloning",
        "toilet_paper": "Toilet Paper",
        "immigration": "Immigration"
    }
    
    if short_or_long_key in special_cases:
        return special_cases[short_or_long_key]
    
    # Default: title case
    key = short_or_long_key.replace("_", " ")
    words = key.split()
    return " ".join(word.capitalize() for word in words)


def _ellipsize(text: str, max_len: int = 90) -> str:
    if len(text) <= max_len:
        return text
    cut = max_len - 1
    # try to cut at last space for nicer look
    space_idx = text.rfind(" ", 0, cut)
    end = space_idx if space_idx > 40 else cut
    return text[:end] + "…"


def get_ab_phrases(long_topic_key: str) -> Tuple[str, str]:
    # Import lazily to avoid circular issues when tools are executed directly
    from network_of_agents.runner import get_topic_framing
    a, b = get_topic_framing(long_topic_key, reversed=False)
    return a, b


def make_descriptive_title(short_topic_key: str, max_len: int = 80, multiline: bool = True) -> str:
    long_key = to_long_topic_key(short_topic_key)
    a, b = get_ab_phrases(long_key)
    a_s = _ellipsize(a, max_len)
    b_s = _ellipsize(b, max_len)
    if multiline:
        return f"{to_display_name(long_key)}\nA: ‘{a_s}’\nvs B: ‘{b_s}’"
    return f"{to_display_name(long_key)} — A: ‘{a_s}’ vs B: ‘{b_s}’"


