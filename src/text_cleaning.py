import re

# --- Cleaning logic (copied from 5_data_clean.py) ---
URL_RE    = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE  = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
HTML_RE   = re.compile(r'<[^>]+>')
HASHTAG   = re.compile(r'#\w+')
MENTION   = re.compile(r'@\w+')


def clean_text(text: str) -> str:
    text = text.lower()
    text = HTML_RE.sub(' ', text)
    text = URL_RE.sub(' ', text)
    text = EMAIL_RE.sub(' ', text)
    text = HASHTAG.sub(' ', text)
    text = MENTION.sub(' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
