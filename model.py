import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from text_preprocessing import preprocess_text

# 1. Veri Yükleme
data = pd.read_csv("mental_health.csv")

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
tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(processed_texts)
sequences = tokenizer.texts_to_sequences(processed_texts)

# Her metnin uzunluğunu hesapla - eşitle ve numpy array formatına çevir
text_lengths = [len(seq) for seq in sequences]
maxlen = int(np.percentile(text_lengths, 97))
padded_sequences = pad_sequences(sequences, padding='post', maxlen=maxlen)
labels = np.array(labels)

# 3. Veri Bölme
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=32)

# 4. Model Oluşturma
model = Sequential([
    Embedding(input_dim=15000, output_dim=128, input_length=maxlen),
    Bidirectional(LSTM(256, return_sequences=True)),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Modeli derle ve özetini göster
model.compile(optimizer=Adam(learning_rate=0.0001, clipvalue=1.0), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Modeli Eğitme
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
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

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Depression'], yticklabels=['Normal', 'Depression'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 7. Modeli - Tokenizeri - Max Lenghti Kaydetme
model.save("depression_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("options.txt", "w") as f:
    f.write(f"maxlen={maxlen}\n")
