import re

def clean_text(text):
    """
    Basic cleaning of the Dreddit dataset.
    Keeping it lightweight and same since it 
    contains useful linguistic signals.
    """

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()