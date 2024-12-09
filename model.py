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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
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
label_counts.index = label_counts.index.map({0: 'Normal', 1: 'Depresyon'})
plt.figure(figsize=(6, 4))
label_counts.plot(kind='bar', color=['blue', 'green'])
plt.title('"Depresyon" ve "Normal" Metinlerin Dağılımı')
plt.xlabel('Etiketler')
plt.ylabel('Sayı')
plt.xticks(rotation=0)
plt.show()

processed_texts = texts.apply(preprocess_text)

# Tokenizer oluştur ve metinleri sayısal verilere çevir
tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
tokenizer.fit_on_texts(processed_texts)
sequences = tokenizer.texts_to_sequences(processed_texts)

vocab_size = len(tokenizer.word_index)
text_lengths = [len(seq) for seq in sequences]
print(f"Veri Kümesindeki Benzersiz Kelimeler: {vocab_size}")

# Her metnin uzunluğunu eşitle ve numpy array formatına çevir
maxlen = int(np.percentile(text_lengths, 98))
padded_sequences = pad_sequences(sequences, padding='post', maxlen=maxlen)
labels = np.array(labels)

# 3. Veri Bölme
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=32, stratify=labels)

# 4. Model Oluşturma
model = Sequential([
    Embedding(input_dim=30000, output_dim=256, input_length=maxlen),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.4),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
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
    batch_size=16,
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Depresyon'], yticklabels=['Normal', 'Depresyon'])
plt.title('Confusion Matrix (Sayısal)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')

plt.subplot(1, 2, 2)
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Normal', 'Depresyon'], yticklabels=['Normal', 'Depresyon'])
plt.title('Confusion Matrix (Normalizasyonlu)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.tight_layout()
plt.show()

# Eğitim ve doğrulama eğrilerinin farkını görselleştirme
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Epochlar Boyunca Kayıp')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Epochlar Boyunca Doğruluk')
plt.legend()
plt.tight_layout()
plt.show()

# ROC Eğrisini ve AUC Skorunu görselleştirme
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred):.2f}')
plt.title('ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.legend()
plt.show()

# 7. Modeli - Tokenizeri - Max Lenghti Kaydetme
model.save("depression_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("options.txt", "w") as f:
    f.write(f"maxlen={maxlen}\n")
