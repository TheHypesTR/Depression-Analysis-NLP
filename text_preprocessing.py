import re
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

spacy.require_gpu()
nlp = spacy.load("en_core_web_md")

# Stop words temizleme
def remove_stopwords(text, custom_stopwords=None):
    stop_words = set(stopwords.words('english'))
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# Lemmatization with POS tagging
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized = []
    for token in doc:
        if token.pos_ not in ['PROPN', 'DET']:
            lemmatized.append(token.lemma_ if token.lemma_ else token.text)
        else:
            lemmatized.append(token.text)
    return ' '.join(lemmatized)

# Metin Temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"@\S+", "USER", text)
    text = re.sub(r"\d+", "NUMBER", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?\.,;]", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "URL", text)
    return text

# Ön işleme
def preprocess_text(text, custom_stopwords=None):
    text = clean_text(text)
    text = remove_stopwords(text, custom_stopwords)
    text = lemmatize_text(text)
    return text
