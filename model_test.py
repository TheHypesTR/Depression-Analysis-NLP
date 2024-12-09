import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text_preprocessing import preprocess_text

def load_maxlen():
    # maxlen değerini options.txt'den yükle
    with open("options.txt", "r") as f:
        for line in f:
            if "maxlen" in line:
                return int(line.strip().split('=')[1])
    raise ValueError("maxlen not found in options.txt")

# 8. Test İşlemi İçin Ayrı Sınıf
def test_model(user_test_texts):
    # Model - Tokenizer - Max Length yükleme
    loaded_model = tf.keras.models.load_model("depression_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        loaded_tokenizer = pickle.load(f)
    maxlen = load_maxlen()

    # Yeni metinleri işleme (ön işleme ve lemmatize etme)
    processed_texts = [preprocess_text(text) for text in user_test_texts]

    # Yeni metinleri sayısal verilere çevir - Tahmin yap
    sequences = loaded_tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=maxlen)
    predictions = loaded_model.predict(padded_sequences)
    
    # Depresyon oranları ve sınıflandırmaları
    prediction_labels = (predictions > 0.5).astype("int32")
    return prediction_labels, predictions

# Test için örnek metinler
user_test_texts = [
    "i hate everything i dont want live more",
    "i feel happy and excited for the day ahead",
    "life feels like a burden i can’t carry anymore",
    "i enjoyed a lovely walk in the park today",
    "i just want to disappear and never come back",
    "i’m looking forward to spending time with my friends",
    "nothing i do seems to matter; i feel so empty",
    "i’m grateful for the little things that make me smile",
    "even surrounded by people, i feel completely alone",
    "today was a productive and fulfilling day"
]

predictions_labels, predictions_probabilities = test_model(user_test_texts)

# Tahminleri ve metinleri yazdırma
for text, label, probability in zip(user_test_texts, predictions_labels, predictions_probabilities):
    depression_probability = probability[0]  # Olasılık değeri
    print(f"Text: {text}\nPrediction: {'Depression' if label == 1 else 'Normal'}")
    print(f"Depression Probability: {depression_probability:.4f}\n")
