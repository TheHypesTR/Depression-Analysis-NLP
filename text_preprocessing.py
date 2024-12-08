import re
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

# Stop words temizleme
def remove_stopwords(text, custom_stopwords=None):
    stop_words = set(stopwords.words('english'))
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# Lemmatization with POS tagging
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ if token.pos_ != 'PROPN' else token.text for token in doc])

# Metin Temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Noktalama ve özel karakterleri kaldır
    text = re.sub(r"\d+", "NUMBER", text)  # Sayıları 'NUMBER' ile değiştir
    return text

# Ön işleme
def preprocess_text(text, custom_stopwords=None):
    text = clean_text(text)
    text = remove_stopwords(text, custom_stopwords)
    text = lemmatize_text(text)
    return text
