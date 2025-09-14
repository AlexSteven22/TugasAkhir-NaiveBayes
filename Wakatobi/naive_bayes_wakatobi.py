import pandas as pd
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(r"E:\IPEN\Semester 7\TA\Simulasi - Augment\tesaurus")

from tesaurus import Tesaurus
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Inisialisasi Stemmer dan Stopword remover
stopword_factory = StopWordRemoverFactory()
stopword_list = stopword_factory.get_stop_words()
stop_words = set(stopword_list)
stemmer = StemmerFactory().create_stemmer()
n=0

# Fungsi augmentasi pada teks asli dengan tesaurus
def augment_tesaurus_n(text, thesaurus, n, max_replace=2):
    augmented_texts = []
    for _ in range(n):
        if not isinstance(text, str) or text.strip() == '':
            continue
        words = text.split()
        new_words = words.copy()
        count = 0
        random.shuffle(words)
        for word in words:
            sinonim = thesaurus.getSinonim(word.lower())
            if sinonim:
                ganti = random.choice(sinonim)
                new_words = [ganti if w.lower() == word.lower() else w for w in new_words]
                count += 1
            if count >= max_replace:
                break
        augmented_text = ' '.join(new_words).strip()
        if augmented_text and augmented_text != text:
            # Terapkan stemming pada teks yang dihasilkan
            augmented_text = stemmer.stem(augmented_text)
            augmented_texts.append(augmented_text)
    return augmented_texts

# Fungsi preprocessing menyeluruh setelah augmentasi
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    stemmed_tokens = [stemmer.stem(w) for w in tokens]
    corrected_tokens = ['rawat' if w == 'awat' else w for w in stemmed_tokens]
    return ' '.join(corrected_tokens)

# Load data train dan test
df_train = pd.read_excel(r'E:\IPEN\Semester 7\TA\Simulasi - Augment\Wakatobi Augment\data_train_wakatobi.xlsx')
df_test = pd.read_excel(r'E:\IPEN\Semester 7\TA\Simulasi - Augment\Wakatobi Augment\data_test_wakatobi.xlsx')

# Validasi kolom
assert 'Review' in df_train.columns and 'Sentimen' in df_train.columns, "Kolom 'Review' dan 'Sentimen' tidak ditemukan di data pelatihan."
assert 'Review' in df_test.columns and 'Sentimen' in df_test.columns, "Kolom 'Review' dan 'Sentimen' tidak ditemukan di data uji."

# Load Tesaurus
tesaurus = Tesaurus(r'E:\IPEN\Semester 7\TA\Simulasi - Augment\tesaurus\dict.json')

# Ambil data negatif untuk augmentasi
df_negatif = df_train[df_train['Sentimen'] == 'Negatif'].copy()
df_non_negatif = df_train[df_train['Sentimen'] != 'Negatif'].copy()
jumlah_positif = len(df_non_negatif)
jumlah_negatif = len(df_negatif)
if jumlah_negatif == 0:
    raise ValueError("Tidak ada data sentimen negatif di data train.")
total_aug_diperlukan = jumlah_positif - jumlah_negatif
n = max(0, total_aug_diperlukan // jumlah_negatif if total_aug_diperlukan % jumlah_negatif != 0 else 0)

augmented_rows = []
augmented_data= []
for idx, row in df_negatif.iterrows():
    original_review = row['Review']
    augmented_texts = augment_tesaurus_n(original_review, tesaurus, n, max_replace=2)
    for aug_text in augmented_texts:
        augmented_rows.append({'Review': aug_text, 'Sentimen': row['Sentimen']})
    while len(augmented_texts) < n:
        augmented_texts.append('')
    row_dict = {'Review Asli': original_review}
    for i in range(n):
        row_dict[f'Hasil Augmentasi {i+1}'] = augmented_texts[i]
    augmented_data.append(row_dict)

df_augmented = pd.DataFrame(augmented_rows)
df_augmented_table = pd.DataFrame(augmented_data)

# Gabungkan data asli dan augmentasi
df_gabungan = pd.concat([df_non_negatif, df_negatif, df_augmented], ignore_index=True)

# Preprocessing seluruh data gabungan
df_gabungan['Review_processed'] = df_gabungan['Review'].apply(preprocess_text)

# Tambahkan kolom 'Review_processed' khusus untuk data augmentasi
df_augmented['Review_processed'] = df_augmented['Review'].apply(preprocess_text)
df_test['Review_processed'] = df_test['Review'].apply(preprocess_text)

# Label encoding untuk train
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df_gabungan['Sentimen'])

# Preprocessing data test
y_test = label_encoder.transform(df_test['Sentimen'])

# Vektorisasi
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df_gabungan['Review_processed'])
X_test = vectorizer.transform(df_test['Review_processed'])

# Training Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Hitung jumlah dan persentase sentimen positif dan negatif yang diprediksi
total_pred = len(y_pred)
pos_pred_count = sum(y_pred == label_encoder.transform(['Positif'])[0])
neg_pred_count = sum(y_pred == label_encoder.transform(['Negatif'])[0])
pos_percentage = (pos_pred_count / total_pred) * 100
neg_percentage = (neg_pred_count / total_pred) * 100

# Hasil prediksi per review (tanpa kolom jumlah/persentase)
df_prediksi = pd.DataFrame({
    'Review': df_test['Review'],
    'Sentimen Asli': label_encoder.inverse_transform(y_test),
    'Prediksi': label_encoder.inverse_transform(y_pred)
})

# Tambahkan baris kosong sebagai pembatas
df_prediksi.loc[len(df_prediksi)] = ['', '', '']

ringkasan = pd.DataFrame({
    'Review': ['SUMMARY'],
    'Sentimen Asli': [f"Total Prediksi: {total_pred}"],
    'Prediksi': [''],
    'Jumlah Positif': [pos_pred_count],
    'Jumlah Negatif': [neg_pred_count],
    'Persentase Positif': [f"{pos_percentage:.2f}%"],
    'Persentase Negatif': [f"{neg_percentage:.2f}%"]
})

# Gabungkan
final_output = pd.concat([df_prediksi, ringkasan], ignore_index=True)

# Visualisasi confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0, digits=4)
print("\nClassification Report:")
print(report)

# Cross validation
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
cv_scores_df = pd.DataFrame(cv_scores, columns=['Akurasi'])
print("\nHasil 10-Fold Cross Validation:")
print(cv_scores_df)
print(f"\nRata-rata Akurasi: {np.mean(cv_scores):.4f}")
print(f"Deviasi Standar Akurasi: {np.std(cv_scores):.4f}")

# Hasil prediksi dataframe
hasil_prediksi = pd.DataFrame({
    'Review': df_test['Review'],
    'Sentimen Asli': label_encoder.inverse_transform(y_test),
    'Prediksi': label_encoder.inverse_transform(y_pred)
})

# Simpan ketiga sheet ke 1 file Excel
output_excel_file = r'E:\IPEN\Semester 7\TA\Simulasi - Augment\Wakatobi Augment\output_augmented_and_predictions.xlsx'

with pd.ExcelWriter(output_excel_file, engine='openpyxl') as writer:
    df_gabungan.to_excel(writer, sheet_name='Train_Augmented', index=False)
    final_output.to_excel(writer, sheet_name='NaiveBayes_Prediction', index=False)
    df_test['Review_processed'].to_excel(writer, sheet_name='Test_Preprocessed', index=False)
    df_augmented.to_excel(writer, sheet_name='Augmented_Preprocessed', index=False)
    df_augmented_table.to_excel(writer, sheet_name='Hasil Augmentasi', index=False)


output_file = r'E:\IPEN\Semester 7\TA\Simulasi - Augment\Wakatobi Augment\hasil_evaluasi_naive_bayes_augmented.txt'
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write("\n\nHasil 10-Fold Cross Validation:\n")
    f.write(cv_scores_df.to_string(index=True))
    f.write(f"\n\nRata-rata Akurasi: {np.mean(cv_scores):.4f}")
    f.write(f"\nDeviasi Standar Akurasi: {np.std(cv_scores):.4f}")

print("\nProses selesai")