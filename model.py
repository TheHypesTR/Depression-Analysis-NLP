import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from text_preprocessing import preprocess_text

# 1. Veri Yükleme
data = pd.read_csv("mental_health.csv")
data = data.dropna(subset=['label'])

# 2. Veri İşleme
# Metin ve etiketleri ayır
texts = data['text']
labels = data['label']

# Grafiği çiz
label_counts = data['label'].value_counts()
label_counts.index = label_counts.index.map({0: 'Normal', 1: 'Depression'})
plt.figure(figsize=(6, 4))
label_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Depression vs Normal Texts Distribution')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

processed_texts = texts.apply(preprocess_text)

# Tokenizer oluştur ve metinleri sayısal verilere çevir
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(processed_texts)
sequences = tokenizer.texts_to_sequences(processed_texts)

vocab_size = len(tokenizer.word_index)
text_lengths = [len(seq) for seq in sequences]
print(f"Unique Words in Dataset: {vocab_size}")

# Uzunluk dağılım grafiği
plt.figure(figsize=(10, 6))
sns.histplot(text_lengths, bins=50, kde=True, color='blue')
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Boxplot ekleyerek daha iyi analiz
plt.figure(figsize=(10, 4))
sns.boxplot(x=text_lengths)
plt.title('Text Length Distribution (Boxplot)')
plt.xlabel('Text Length')
plt.show()

# Her metnin uzunluğunu eşitle ve numpy array formatına çevir
maxlen = int(np.percentile(text_lengths, 95))
padded_sequences = pad_sequences(sequences, padding='post', maxlen=maxlen)
labels = np.array(labels)

sns.boxplot(x=text_lengths)
plt.title("Text Length Distribution")
plt.show()

# 3. Veri Bölme
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=32)

# 4. Model Oluşturma
model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=maxlen),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Modeli derle ve özetini göster
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Modeli Eğitme
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=16,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stopping],
    verbose=1
)

# 6. Modeli Değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Tahmin ve Raporlama
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Tahmin ve Normalizasyonlu ve Normal confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Depression'], yticklabels=['Normal', 'Depression'])
plt.title('Confusion Matrix (Counts)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Normal', 'Depression'], yticklabels=['Normal', 'Depression'])
plt.title('Confusion Matrix (Normalized)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()

# Eğitim ve doğrulama eğrilerinin farkını görselleştirme
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
difference = np.array(history.history['val_loss']) - np.array(history.history['loss'])
plt.plot(difference, label='Validation Loss - Train Loss', color='purple')
plt.axhline(0, linestyle='--', color='gray', linewidth=0.8)
plt.title('Validation - Train Loss Difference')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# 7. Modeli - Tokenizeri - Max Lenghti Kaydetme
model.save("depression_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("options.txt", "w") as f:
    f.write(f"maxlen={maxlen}\n")
