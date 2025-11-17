import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# NLTK optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Try to ensure basic nltk data (silently)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception:
    pass


class ElectronicsPreprocessor:
    """
    Preprocessor untuk dataset review elektronik + fitur TF-IDF dan KNN index.
    Dirancang hemat memori (tanpa full similarity matrix default).
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.df_processed = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.cosine_sim_matrix = None  # optional fallback
        self.knn = None

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set([
                'the','a','an','and','or','of','to','for','in','on','with','is','it','this','that','by','from','at','as','are','be','was','were'
            ])

    # -------------------- Load & Clean --------------------
    def load_data(self):
        self.df = pd.read_excel(self.file_path)
        return self.df

    def clean_data(self):
        if self.df is None:
            raise RuntimeError('Data belum dimuat')
        df = self.df.copy()

        # Normalize IDs to string
        for id_col in ['customer_id', 'product_id']:
            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str)

        # Fill text
        for col in ['product_title', 'review_headline', 'review_body']:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # Numeric
        for col in ['star_rating', 'helpful_votes', 'total_votes']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Date
        if 'review_date' in df.columns:
            df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')

        # Valid rating 1..5
        if 'star_rating' in df.columns:
            df = df[(df['star_rating'] >= 1) & (df['star_rating'] <= 5)]

        self.df_processed = df.reset_index(drop=True)
        return self.df_processed

    # -------------------- Text utils --------------------
    def _clean_text(self, text):
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text

    def _tokenize(self, text):
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def _remove_stop(self, tokens):
        return [w for w in tokens if w not in self.stop_words and len(w) > 2]

    def _lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(w) for w in tokens]

    # -------------------- Text preprocessing --------------------
    def preprocess_text(self, use_stemming=False):
        if self.df_processed is None:
            raise RuntimeError('Data belum dibersihkan')
        df = self.df_processed

        # Ensure columns exist & string
        for col in ['product_title', 'review_headline', 'review_body']:
            if col not in df.columns:
                df[col] = ''
            df[col] = df[col].astype('string').fillna('')

        df['combined_text'] = (df['product_title'] + ' ' + df['review_headline'] + ' ' + df['review_body']).str.strip()
        df['cleaned_text'] = df['combined_text'].apply(self._clean_text)
        df['tokens'] = df['cleaned_text'].apply(self._tokenize)
        df['tokens_no_stop'] = df['tokens'].apply(self._remove_stop)
        df['processed_tokens'] = df['tokens_no_stop'].apply(self._lemmatize) if not use_stemming else df['tokens_no_stop'].apply(lambda t: [self.stemmer.stem(w) for w in t])
        df['processed_text'] = df['processed_tokens'].apply(lambda t: ' '.join(t))

        # Drop empty
        df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)

        # Basic features
        df['text_length'] = df['combined_text'].str.len()
        df['word_count'] = df['processed_tokens'].apply(len)
        df['char_count'] = df['processed_text'].str.len()
        df['is_positive'] = (df['star_rating'] >= 4).astype(int) if 'star_rating' in df.columns else 0
        df['is_negative'] = (df['star_rating'] <= 2).astype(int) if 'star_rating' in df.columns else 0
        df['helpful_ratio'] = np.where(df.get('total_votes', 0).fillna(0) > 0,
                                       df.get('helpful_votes', 0).fillna(0) / df.get('total_votes', 1).replace(0, 1),
                                       0)

        # Product aggregations (if available)
        if 'product_id' in df.columns:
            agg = df.groupby('product_id').agg({
                'star_rating': ['mean', 'std', 'count'],
                'helpful_votes': 'sum',
                'total_votes': 'sum'
            }).reset_index()
            agg.columns = ['product_id', 'product_avg_rating', 'product_rating_std', 'product_review_count', 'product_total_helpful', 'product_total_votes']
            df = df.merge(agg, on='product_id', how='left')

        self.df_processed = df
        return df

    # -------------------- TF-IDF & Similarity --------------------
    def create_tfidf_matrix(self, max_features=5000, ngram_range=(1, 2)):
        if self.df_processed is None or 'processed_text' not in self.df_processed.columns:
            raise RuntimeError('Text belum diproses')
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df_processed['processed_text'])
        return self.tfidf_matrix

    def build_knn_index(self, n_neighbors=50):
        if self.tfidf_matrix is None:
            raise RuntimeError('TF-IDF belum dibuat')
        from sklearn.neighbors import NearestNeighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute', n_jobs=-1)
        self.knn.fit(self.tfidf_matrix)
        return self.knn

    # Optional heavy path, not used by default
    def create_similarity_matrix(self, metric='cosine'):
        if self.tfidf_matrix is None:
            raise RuntimeError('TF-IDF belum dibuat')
        if metric == 'cosine':
            self.cosine_sim_matrix = cosine_similarity(self.tfidf_matrix)
        else:
            dist = euclidean_distances(self.tfidf_matrix)
            self.cosine_sim_matrix = 1 / (1 + dist)
        return self.cosine_sim_matrix

    # -------------------- Analysis --------------------
    def analyze_data(self):
        if self.df_processed is None:
            raise RuntimeError('Data belum diproses')
        df = self.df_processed
        sentiment = {
            'positive': int((df.get('star_rating', pd.Series()).ge(4)).sum()) if 'star_rating' in df.columns else 0,
            'neutral': int((df.get('star_rating', pd.Series()).eq(3)).sum()) if 'star_rating' in df.columns else 0,
            'negative': int((df.get('star_rating', pd.Series()).le(2)).sum()) if 'star_rating' in df.columns else 0,
        }
        return {
            'total_products': int(df.get('product_id', pd.Series()).nunique()) if 'product_id' in df.columns else 0,
            'total_reviews': int(len(df)),
            'avg_rating': float(df.get('star_rating', pd.Series(dtype=float)).mean()) if 'star_rating' in df.columns else 0.0,
            'sentiment': sentiment,
        }

    # -------------------- Pipeline --------------------
    def run_full_pipeline(self, use_stemming=False, save_output=False):
        self.load_data()
        self.clean_data()
        self.preprocess_text(use_stemming=use_stemming)
        self.create_tfidf_matrix(max_features=5000, ngram_range=(1, 2))
        self.build_knn_index(n_neighbors=50)
        analysis = self.analyze_data()
        return {
            'processed_data': self.df_processed,
            'tfidf_matrix': self.tfidf_matrix,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'similarity_matrix': self.cosine_sim_matrix,
            'analysis': analysis,
        }

    # -------------------- Recommendations --------------------
    def get_recommendations(self, product_id=None, product_title=None, top_n=10):
        if self.knn is None and self.cosine_sim_matrix is None:
            raise RuntimeError('Model kesamaan belum dibuat')
        df = self.df_processed
        if product_id is not None:
            idxs = df.index[df['product_id'] == str(product_id)]
        elif product_title is not None:
            idxs = df.index[df['product_title'].str.contains(product_title, case=False, na=False)]
        else:
            raise ValueError('Berikan product_id atau product_title')
        if len(idxs) == 0:
            raise ValueError('Produk tidak ditemukan')
        idx = idxs[0]

        if self.knn is not None:
            distances, indices = self.knn.kneighbors(self.tfidf_matrix[idx], n_neighbors=top_n+1)
            indices = indices.ravel()[1:]
            sims = 1 - distances.ravel()[1:]
            recs = df.iloc[indices].copy()
            recs['similarity_score'] = sims
            recs['rank'] = range(1, len(recs) + 1)
            return recs
        else:
            sims = list(enumerate(self.cosine_sim_matrix[idx]))
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:top_n+1]
            indices = [i for i, _ in sims]
            recs = df.iloc[indices].copy()
            recs['similarity_score'] = [s for _, s in sims]
            recs['rank'] = range(1, len(recs) + 1)
            return recs
