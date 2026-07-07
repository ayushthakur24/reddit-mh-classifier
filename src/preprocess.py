import re


def clean_text(text):
    """
    Cleans Reddit posts before passing them to the ML model.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove punctuation
    4. Remove extra spaces
    """

    # Convert input to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    # Remove multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()