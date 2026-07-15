"""
Unit tests for the preprocessing module.
"""

from src.preprocess import clean_text


# ==========================================================
# Lowercase Conversion
# ==========================================================

def test_lowercase_conversion():

    text = "HELLO WORLD"

    assert clean_text(text) == "hello world"


# ==========================================================
# Remove Punctuation
# ==========================================================

def test_remove_punctuation():

    text = "Hello!!! How are you???"

    assert clean_text(text) == "hello how are you"


# ==========================================================
# Remove URL
# ==========================================================

def test_remove_url():

    text = "Visit https://google.com now"

    assert clean_text(text) == "visit now"


# ==========================================================
# Remove Extra Spaces
# ==========================================================

def test_remove_extra_spaces():

    text = "Hello      World"

    assert clean_text(text) == "hello world"


# ==========================================================
# Empty String
# ==========================================================

def test_empty_string():

    assert clean_text("") == ""


# ==========================================================
# Numbers Should Remain
# ==========================================================

def test_numbers_remain():

    text = "I have 2 dogs"

    assert clean_text(text) == "i have 2 dogs"