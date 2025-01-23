import nltk
from nltk.corpus import stopwords
import re

# Unduh stopwords jika belum tersedia
try:
    from __main__ import stop_words
except ImportError:
    # Fallback jika import gagal
    import os
    
    def load_stopwords(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return set(file.read().splitlines())
        except Exception:
            return set()

    stopwords_path = 'stopwords/indonesian'
    stop_words = load_stopwords(stopwords_path) if os.path.exists(stopwords_path) else set()


def preprocess_text(text):
    """
    Melakukan preprocessing teks: lowercasing, stop words removal, dan tokenization.
    """
    # Lowercasing
    text = text.lower()
    # Menghapus karakter non-alphabet
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization
    tokens = text.split()
    # Menghapus stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Gabungkan kembali menjadi teks
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text
