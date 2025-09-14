import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi stemmer dan stopword remover
stemmer = StemmerFactory().create_stemmer()
stopword_factory = StopWordRemoverFactory()
stop_words = set(stopword_factory.get_stop_words())

# Fungsi preprocessing
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    stemmed_tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(stemmed_tokens)

# Load data
file_path = r'E:\IPEN\Semester 7\TA\Simulasi - Augment\Tanjung Kelayang Augment\data_kelayang - visualisasi.xlsx'
df = pd.read_excel(file_path, sheet_name=0)
df = df[['Review', 'Sentimen']].dropna()

# Preprocessing kolom Review
df['Review_processed'] = df['Review'].apply(preprocess_text)

# Tambahkan daftar stopwords Bahasa Indonesia
custom_stopwords = {
    'pulau', 'sangat', 'sekali', 'tidak', 'ada', 'itu','ini','dan', 'yang', 'di', 'ke', 
    'dari', 'dengan', 'untuk', 'pada', 'dalam', 'juga', 'karena', 'banyak', 
    'tempat', 'lokasi', 'area', 'kami', 'sudah', 'hanya', 'terlalu', 'bisa', 
    'jadi', 'seperti', 'cukup', 'lebih', 'kurang', 'kelayang', 'wisata', 'pantai', 'tapi',
    'orang','perahu','belitung','sayang','masih','agak','gk','tanjung','lengkuas',
    'atau','harus','salah','satu','anda','kita','kami','saya','kamu','nya','sini'
}

# Gabungkan dengan stopwords default
stopwords_total = STOPWORDS.union(custom_stopwords)

# Gabungkan review berdasarkan sentimen
positif = ' '.join(df[df['Sentimen'] == 'Positif']['Review_processed'].astype(str))
negatif = ' '.join(df[df['Sentimen'] == 'Negatif']['Review_processed'].astype(str))

# Buat WordCloud untuk sentimen positif
wordcloud_positif = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Greens',
    stopwords=stopwords_total
).generate(positif)

# Buat WordCloud untuk sentimen negatif
wordcloud_negatif = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Reds',
    stopwords=stopwords_total
).generate(negatif)

# Tampilkan WordCloud dalam 1 figure terpisah
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.title("WordCloud - Sentimen Positif", pad=20)
plt.imshow(wordcloud_positif, interpolation='bilinear')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("WordCloud - Sentimen Negatif", pad=20)
plt.imshow(wordcloud_negatif, interpolation='bilinear')
plt.axis('off')

plt.tight_layout()
plt.savefig(r"E:\IPEN\Semester 7\TA\Simulasi - Augment\Tanjung Kelayang Augment\wordcloud_kelayang.png", 
            dpi=300, 
            bbox_inches='tight')
plt.show()