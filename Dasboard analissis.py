import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Electronics AI Recommendation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ENHANCED CSS WITH WHITE BACKGROUND PROFESSIONAL THEME ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background - Pure White */
    .main {
        background: #FFFFFF;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Hero Header - Professional White with Gold Accents */
    .hero-header {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        position: relative;
        overflow: hidden;
        border: 3px solid #FFD700;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 215, 0, 0.05) 0%, transparent 70%);
        animation: pulse 15s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1) translate(0, 0); }
        50% { transform: scale(1.1) translate(5%, 5%); }
    }
    
    .hero-header h1 {
        color: #1A1A1A;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 1;
    }
    
    .hero-header p {
        color: #555555;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    
    .badge {
        background: #FFD700;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #1A1A1A;
        font-size: 0.9rem;
        font-weight: 700;
        border: 2px solid #FFC700;
        box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3);
    }
    
    /* Modern Metric Cards - White Professional */
    .metric-card {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 2px solid #E0E0E0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(255, 215, 0, 0.3);
        border-color: #FFD700;
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1A1A1A;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-change {
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    .metric-change.positive {
        color: #FFD700;
    }
    
    /* Enhanced Product Card - White Professional */
    .product-card {
        background: #FFFFFF;
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.2rem;
        border-left: 5px solid #FFD700;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .product-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 30px rgba(255, 215, 0, 0.2);
        border-left-color: #FFC700;
    }
    
    .product-rank {
        position: absolute;
        top: 1rem;
        right: 1rem;
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #FFD700 0%, #FFC700 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #1A1A1A;
        font-weight: 800;
        font-size: 1.2rem;
        box-shadow: 0 4px 10px rgba(255, 215, 0, 0.4);
    }
    
    .product-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1A1A1A;
        margin-bottom: 0.8rem;
        padding-right: 50px;
        line-height: 1.4;
    }
    
    .product-rating {
        color: #FFD700;
        font-size: 1.1rem;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    
    .product-meta {
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .meta-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        color: #666666;
        font-size: 0.9rem;
    }
    
    .similarity-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: linear-gradient(135deg, #FFD700 0%, #FFC700 100%);
        color: #1A1A1A;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
        margin-top: 1rem;
    }
    
    /* Algorithm Badge */
    .algorithm-badge {
        background: linear-gradient(135deg, #FFD700 0%, #FFC700 100%);
        color: #1A1A1A;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Info Box */
    .info-box {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #FFD700;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .info-box h4 {
        color: #1A1A1A;
        margin: 0 0 0.5rem 0;
        font-weight: 700;
    }
    
    .info-box p {
        color: #555555;
    }
    
    /* Search Container */
    .search-container {
        background: #FFFFFF;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 2px solid #E0E0E0;
    }
    
    /* Search Input Styling */
    .search-input {
        background: #F8F9FA;
        border: 2px solid #E0E0E0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s;
        color: #1A1A1A;
    }
    
    .search-input:focus {
        border-color: #FFD700;
        box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.1);
        outline: none;
    }
    
    /* Sidebar Enhancement - Professional Gray */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8F9FA 0%, #E8E9EB 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #1A1A1A !important;
    }
    
    /* Button Styling - GOLD PRIMARY */
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFC700 100%);
        color: #1A1A1A;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.4);
        background: linear-gradient(135deg, #FFC700 0%, #FFD700 100%);
    }
    
    /* Tab Styling - Professional */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #F8F9FA;
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        border: 2px solid #E0E0E0;
        color: #666666;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFC700 100%);
        color: #1A1A1A;
        border-color: #FFD700;
    }
    
    /* Text colors - Professional */
    h1, h2, h3, h4, h5, h6 {
        color: #1A1A1A !important;
    }
    
    p, span, div {
        color: #333333;
    }
    
    /* Stats box */
    .stats-box {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #E0E0E0;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Feature card */
    .feature-card {
        background: #FFFFFF;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #FFD700;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        transition: all 0.3s;
    }
    
    .feature-card:hover {
        transform: translateX(5px);
        border-left-color: #FFC700;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
    }
    
    .feature-card b {
        color: #1A1A1A;
    }
    
    .feature-card span {
        color: #666666;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background: #F8F9FA;
        color: #1A1A1A;
        border: 2px solid #E0E0E0;
        border-radius: 10px;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #FFD700;
        box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.1);
    }
    
    /* Select boxes */
    .stSelectbox>div>div {
        background: #F8F9FA;
        border: 2px solid #E0E0E0;
        border-radius: 10px;
    }
    
    /* Sliders */
    .stSlider>div>div>div {
        background: #FFD700;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1A1A1A;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 2px solid #E0E0E0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========== ENHANCED PREPROCESSOR CLASS ==========
class AdvancedElectronicsRecommender:
    def __init__(self, df):
        self.df = df
        self.tfidf_matrix = None
        self.svd_matrix = None
        self.vectorizer = None
        self.svd_model = None
        self.clusters = None
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        import re
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def build_content_based_model(self, use_svd=True, n_components=100):
        """Build advanced content-based filtering model"""
        self.df['combined_text'] = (
            self.df['product_title'].fillna('') + ' ' +
            self.df['review_headline'].fillna('') + ' ' +
            self.df['processed_text'].fillna('')
        )
        
        self.df['combined_text'] = self.df['combined_text'].apply(self.preprocess_text)
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_text'])
        
        if use_svd and self.tfidf_matrix.shape[1] > n_components:
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd_matrix = self.svd_model.fit_transform(self.tfidf_matrix)
            return "TF-IDF + SVD (Latent Semantic Analysis)"
        else:
            self.svd_matrix = self.tfidf_matrix.toarray()
            return "TF-IDF"
    
    def build_collaborative_features(self):
        """Add collaborative filtering features"""
        product_stats = self.df.groupby('product_id').agg({
            'star_rating': ['mean', 'std', 'count'],
            'helpful_votes': 'sum',
            'total_votes': 'sum'
        }).reset_index()
        
        product_stats.columns = ['product_id', 'avg_rating', 'rating_std', 'review_count', 
                                  'total_helpful', 'total_votes']
        
        product_stats['helpful_ratio'] = np.where(
            product_stats['total_votes'] > 0,
            product_stats['total_helpful'] / product_stats['total_votes'],
            0
        )
        
        self.df = self.df.merge(product_stats, on='product_id', how='left')
        
        return product_stats
    
    def perform_clustering(self, n_clusters=5):
        """Perform K-Means clustering on products"""
        if self.svd_matrix is not None:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.df['cluster'] = kmeans.fit_predict(self.svd_matrix)
            self.clusters = kmeans
            return kmeans
        return None
    
    def sentiment_analysis_advanced(self):
        """Advanced sentiment analysis with statistical insights"""
        sentiment_data = {
            'positive': (self.df['star_rating'] >= 4).sum(),
            'neutral': (self.df['star_rating'] == 3).sum(),
            'negative': (self.df['star_rating'] <= 2).sum(),
            'highly_positive': (self.df['star_rating'] == 5).sum(),
            'highly_negative': (self.df['star_rating'] == 1).sum(),
        }
        
        # Statistical measures
        sentiment_data['mean_rating'] = self.df['star_rating'].mean()
        sentiment_data['median_rating'] = self.df['star_rating'].median()
        sentiment_data['std_rating'] = self.df['star_rating'].std()
        sentiment_data['skewness'] = self.df['star_rating'].skew()
        
        return sentiment_data
    
    def get_hybrid_recommendations(self, product_id=None, product_title=None, top_n=10, 
                                   content_weight=0.7, rating_weight=0.3):
        """Get hybrid recommendations"""
        if product_title:
            mask = self.df['product_title'] == product_title
        elif product_id:
            mask = self.df['product_id'] == product_id
        else:
            return None
        
        if mask.sum() == 0:
            return None
        
        idx = mask.idxmax()
        
        content_sim = cosine_similarity([self.svd_matrix[idx]], self.svd_matrix)[0]
        
        target_rating = self.df.loc[idx, 'avg_rating']
        rating_sim = 1 - np.abs(self.df['avg_rating'] - target_rating) / 5
        
        hybrid_score = (content_weight * content_sim) + (rating_weight * rating_sim)
        
        similar_indices = hybrid_score.argsort()[::-1][1:top_n+1]
        
        recommendations = self.df.iloc[similar_indices].copy()
        recommendations['similarity_score'] = hybrid_score[similar_indices]
        recommendations['content_similarity'] = content_sim[similar_indices]
        recommendations['rating_similarity'] = rating_sim[similar_indices]
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        return recommendations
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        corr_features = ['star_rating', 'helpful_votes', 'total_votes', 
                        'text_length', 'word_count']
        corr_features = [f for f in corr_features if f in self.df.columns]
        
        if len(corr_features) > 1:
            return self.df[corr_features].corr()
        return None

# ========== SESSION STATE ==========
if 'df' not in st.session_state:
    st.session_state.df = None
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'algorithm_used' not in st.session_state:
    st.session_state.algorithm_used = None

# ========== ENHANCED VISUALIZATION FUNCTIONS ==========
def create_advanced_rating_chart(df):
    """Create advanced rating distribution"""
    rating_counts = df['star_rating'].value_counts().sort_index()
    
    fig = go.Figure()
    
    colors = ['#E74C3C', '#E67E22', '#F39C12', '#ffd700', '#00bfff']
    
    for rating, count in rating_counts.items():
        fig.add_trace(go.Bar(
            x=[rating],
            y=[count],
            name=f'{rating}⭐',
            marker=dict(
                color=colors[rating-1],
                line=dict(color='#1a1a2e', width=2)
            ),
            text=count,
            textposition='outside',
            textfont=dict(color='#e0e0e0'),
            hovertemplate=f'<b>{rating} Stars</b><br>Count: {count:,}<br>Percentage: {count/len(df)*100:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': '⭐ Rating Distribution Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00bfff', 'family': 'Inter'}
        },
        xaxis_title='Star Rating',
        yaxis_title='Number of Reviews',
        template='plotly_dark',
        height=450,
        showlegend=False,
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0')
    )
    
    return fig

def create_sentiment_donut(df):
    """Create modern donut chart"""
    positive = (df['star_rating'] >= 4).sum()
    neutral = (df['star_rating'] == 3).sum()
    negative = (df['star_rating'] <= 2).sum()
    
    labels = ['😊 Positive', '😐 Neutral', '😞 Negative']
    values = [positive, neutral, negative]
    colors = ['#00bfff', '#ffd700', '#E74C3C']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='#1a1a2e', width=3)),
        textinfo='label+percent',
        textfont=dict(size=14, family='Inter', color='#1a1a2e'),
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>',
        pull=[0.1, 0, 0]
    )])
    
    fig.add_annotation(
        text=f"<b>{len(df):,}</b><br>Reviews",
        x=0.5, y=0.5,
        font=dict(size=20, family='Inter', color='#00bfff'),
        showarrow=False
    )
    
    fig.update_layout(
        title={
            'text': '💭 Sentiment Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00bfff', 'family': 'Inter'}
        },
        height=450,
        showlegend=True,
        template='plotly_dark',
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0')
    )
    
    return fig

def create_cluster_visualization(df):
    """Create cluster visualization"""
    if 'cluster' not in df.columns:
        return None
    
    cluster_stats = df.groupby('cluster').agg({
        'star_rating': 'mean',
        'product_id': 'count',
        'helpful_votes': 'sum',
        'product_title': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    
    cluster_stats.columns = ['cluster', 'avg_rating', 'product_count', 'total_helpful', 'representative_product']
    
    fig = go.Figure()
    
    # Bar chart untuk product count
    fig.add_trace(go.Bar(
        x=cluster_stats['cluster'],
        y=cluster_stats['product_count'],
        name='Product Count',
        marker=dict(
            color=cluster_stats['avg_rating'],
            colorscale=[[0, '#E74C3C'], [0.5, '#FFD700'], [1, '#4CAF50']],
            showscale=True,
            colorbar=dict(
                title='Avg Rating',
                titlefont=dict(color='#333333'),
                tickfont=dict(color='#333333')
            ),
            line=dict(color='#333333', width=2)
        ),
        text=[f"{count}<br>{rating:.2f}⭐" for count, rating in zip(cluster_stats['product_count'], cluster_stats['avg_rating'])],
        textposition='outside',
        textfont=dict(color='#333333', size=12, family='Inter'),
        hovertemplate='<b>Cluster %{x}</b><br>Products: %{y}<br>Avg Rating: %{marker.color:.2f}⭐<br>Total Helpful: %{customdata}<extra></extra>',
        customdata=cluster_stats['total_helpful']
    ))
    
    fig.update_layout(
        title={
            'text': '🎯 Product Clustering Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#333333', 'family': 'Inter', 'weight': 700}
        },
        xaxis_title='Cluster ID',
        yaxis_title='Number of Products',
        template='plotly_white',
        height=450,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#F8F9FA',
        font=dict(color='#333333', family='Inter'),
        showlegend=False,
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            gridcolor='#E0E0E0'
        ),
        yaxis=dict(
            gridcolor='#E0E0E0'
        )
    )
    
    return fig

def create_correlation_heatmap(corr_matrix):
    """Create correlation heatmap"""
    if corr_matrix is None:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[[0, '#E74C3C'], [0.5, '#2c3e50'], [1, '#00bfff']],
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12, "color": "#e0e0e0"},
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(
        title={
            'text': '🔗 Feature Correlation Matrix',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00bfff'}
        },
        template='plotly_dark',
        height=500,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0')
    )
    
    return fig

def create_time_series_chart(df):
    """Create time series analysis"""
    if 'review_date' not in df.columns:
        return None
    
    df_temp = df.copy()
    df_temp['year_month'] = df_temp['review_date'].dt.to_period('M').astype(str)
    temporal_data = df_temp.groupby('year_month').agg({
        'product_id': 'count',
        'star_rating': 'mean'
    }).rename(columns={'product_id': 'review_count'})
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=temporal_data.index,
            y=temporal_data['review_count'],
            name='Review Count',
            marker_color='#00bfff',
            hovertemplate='<b>%{x}</b><br>Reviews: %{y:,}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=temporal_data.index,
            y=temporal_data['star_rating'],
            name='Avg Rating',
            mode='lines+markers',
            line=dict(color='#ffd700', width=3),
            marker=dict(size=8, color='#ffd700'),
            hovertemplate='<b>%{x}</b><br>Avg Rating: %{y:.2f}⭐<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title={
            'text': '📈 Temporal Trend Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00bfff'}
        },
        template='plotly_dark',
        height=450,
        hovermode='x unified',
        xaxis=dict(tickangle=-45),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0')
    )
    
    fig.update_yaxes(title_text="Number of Reviews", secondary_y=False, color='#00bfff')
    fig.update_yaxes(title_text="Average Rating", secondary_y=True, color='#ffd700')
    
    return fig

def create_box_plot(df):
    """Create box plot for rating distribution"""
    fig = go.Figure()
    
    colors = ['#E74C3C', '#E67E22', '#F39C12', '#ffd700', '#00bfff']
    
    for i, rating in enumerate(sorted(df['star_rating'].unique())):
        subset = df[df['star_rating'] == rating]['helpful_votes']
        fig.add_trace(go.Box(
            y=subset,
            name=f'{rating}⭐',
            marker_color=colors[i],
            line=dict(color=colors[i]),
            boxmean='sd'
        ))
    
    fig.update_layout(
        title={
            'text': '📦 Helpful Votes Distribution by Rating',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00bfff'}
        },
        yaxis_title='Helpful Votes',
        xaxis_title='Star Rating',
        template='plotly_dark',
        height=450,
        showlegend=True,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0')
    )
    
    return fig

def create_violin_plot(df):
    """Create violin plot for word count distribution"""
    fig = go.Figure()
    
    colors = ['#E74C3C', '#E67E22', '#F39C12', '#ffd700', '#00bfff']
    
    for i, rating in enumerate(sorted(df['star_rating'].unique())):
        subset = df[df['star_rating'] == rating]['word_count']
        fig.add_trace(go.Violin(
            y=subset,
            name=f'{rating}⭐',
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[i],
            line_color=colors[i],
            opacity=0.7
        ))
    
    fig.update_layout(
        title={
            'text': '🎻 Word Count Distribution by Rating',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00bfff'}
        },
        yaxis_title='Word Count',
        xaxis_title='Star Rating',
        template='plotly_dark',
        height=450,
        showlegend=True,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0')
    )
    
    return fig

def create_scatter_3d(df):
    """Create 3D scatter plot"""
    sample_df = df.sample(min(1000, len(df)))
    
    fig = go.Figure(data=[go.Scatter3d(
        x=sample_df['star_rating'],
        y=sample_df['helpful_votes'],
        z=sample_df['word_count'],
        mode='markers',
        marker=dict(
            size=5,
            color=sample_df['star_rating'],
            colorscale=[[0, '#E74C3C'], [0.5, '#ffd700'], [1, '#00bfff']],
            showscale=True,
            colorbar=dict(title='Rating'),
            line=dict(color='#1a1a2e', width=0.5)
        ),
        text=sample_df['product_title'],
        hovertemplate='<b>%{text}</b><br>Rating: %{x}<br>Helpful: %{y}<br>Words: %{z}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': '🌐 3D Feature Space Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00bfff'}
        },
        scene=dict(
            xaxis=dict(title='Rating', backgroundcolor='#1a1a2e', gridcolor='#2c3e50'),
            yaxis=dict(title='Helpful Votes', backgroundcolor='#1a1a2e', gridcolor='#2c3e50'),
            zaxis=dict(title='Word Count', backgroundcolor='#1a1a2e', gridcolor='#2c3e50'),
        ),
        template='plotly_dark',
        height=600,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0')
    )
    
    return fig

def create_funnel_chart(df):
    """Create funnel chart for rating distribution"""
    rating_counts = df['star_rating'].value_counts().sort_index(ascending=False)
    
    fig = go.Figure(go.Funnel(
        y=[f'{r} Stars' for r in rating_counts.index],
        x=rating_counts.values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=['#00bfff', '#ffd700', '#F39C12', '#E67E22', '#E74C3C']
        ),
        connector=dict(line=dict(color='#1a1a2e', width=3))
    ))
    
    fig.update_layout(
        title={
            'text': '🔻 Rating Funnel Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00bfff'}
        },
        template='plotly_dark',
        height=500,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0')
    )
    
    return fig

def create_sunburst_chart(df):
    """Create sunburst chart for hierarchical analysis"""
    if 'verified_purchase' not in df.columns:
        return None
    
    # Create rating categories
    df_temp = df.copy()
    df_temp['rating_category'] = pd.cut(
        df_temp['star_rating'], 
        bins=[0, 2, 3, 5], 
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    df_temp['verified_category'] = df_temp['verified_purchase'].map({
        'Y': 'Verified', 
        'N': 'Unverified',
        'y': 'Verified',
        'n': 'Unverified'
    }).fillna('Unknown')
    
    # Remove any NaN values
    df_temp = df_temp.dropna(subset=['rating_category', 'verified_category'])
    
    if len(df_temp) == 0:
        return None
    
    # Aggregate data
    agg_data = df_temp.groupby(['rating_category', 'verified_category']).size().reset_index(name='count')
    
    # Prepare data for sunburst
    labels = []
    parents = []
    values = []
    colors_list = []
    
    # Root level - rating categories
    rating_totals = agg_data.groupby('rating_category')['count'].sum()
    color_map = {'Negative': '#E74C3C', 'Neutral': '#ffd700', 'Positive': '#00bfff'}
    
    for rating in ['Negative', 'Neutral', 'Positive']:
        if rating in rating_totals.index:
            labels.append(rating)
            parents.append('')
            values.append(rating_totals[rating])
            colors_list.append(color_map[rating])
    
    # Second level - verified status
    verified_color_map = {
        'Verified': {'Negative': '#C0392B', 'Neutral': '#F39C12', 'Positive': '#0080ff'},
        'Unverified': {'Negative': '#E67E22', 'Neutral': '#FFD93D', 'Positive': '#4ECDC4'},
        'Unknown': {'Negative': '#95A5A6', 'Neutral': '#BDC3C7', 'Positive': '#7F8C8D'}
    }
    
    for _, row in agg_data.iterrows():
        rating = row['rating_category']
        verified = row['verified_category']
        count = row['count']
        
        label = f"{verified}"
        labels.append(label)
        parents.append(rating)
        values.append(count)
        colors_list.append(verified_color_map.get(verified, {}).get(rating, '#95A5A6'))
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors_list, line=dict(color='#1a1a2e', width=2)),
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '☀️ Hierarchical Review Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#00bfff'}
        },
        template='plotly_dark',
        height=500,
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0')
    )
    
    return fig

def display_enhanced_product_card(row, rank):
    """Display enhanced product card with better visibility"""
    # Create container with custom styling
    st.markdown(f"""
    <div style="background: #FFFFFF; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); 
                margin-bottom: 1rem; border-left: 5px solid #FFD700; position: relative;">
        <div style="position: absolute; top: 1rem; right: 1rem; background: linear-gradient(135deg, #FFD700, #FFC700); 
                    padding: 0.5rem 1rem; border-radius: 20px; font-weight: 700; color: #1A1A1A; box-shadow: 0 4px 10px rgba(255,215,0,0.3);">
            #{rank} • {row['similarity_score']*100:.1f}% Match
        </div>
        
        <h3 style="color: #1A1A1A; margin-bottom: 0.8rem; padding-right: 150px; font-weight: 700;">
            {row['product_title'][:150]}
        </h3>
        
        <div style="color: #FFD700; font-size: 1.1rem; margin: 0.5rem 0; font-weight: 600;">
            {"⭐" * int(row['star_rating'])} <span style="color: #1A1A1A;">({row['star_rating']:.1f}/5.0)</span>
        </div>
        
        <div style="display: flex; gap: 2rem; margin-top: 1rem; flex-wrap: wrap; color: #333333;">
            <div style="display: flex; align-items: center; gap: 0.4rem;">
                <span>📝</span>
                <span style="color: #1A1A1A; font-weight: 600;">{int(row.get('review_count', 0)) if not pd.isna(row.get('review_count', 0)) else 'N/A'}</span>
                <span style="color: #666666;">reviews</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.4rem;">
                <span>👍</span>
                <span style="color: #1A1A1A; font-weight: 600;">{int(row.get('total_helpful', 0))}</span>
                <span style="color: #666666;">helpful votes</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.4rem;">
                <span>📊</span>
                <span style="color: #1A1A1A; font-weight: 600;">{row.get('helpful_ratio', 0):.1%}</span>
                <span style="color: #666666;">helpful ratio</span>
            </div>
        </div>
        
        <div style="margin-top: 1rem; padding: 0.8rem; background: #F8F9FA; border-radius: 10px; border: 2px solid #E0E0E0;">
            <div style="color: #1A1A1A; font-weight: 600; margin-bottom: 0.3rem;">🎯 Similarity Breakdown:</div>
            <div style="color: #333333;">
                Content Similarity: <span style="color: #FFD700; font-weight: 700;">{row.get('content_similarity', 0)*100:.1f}%</span> | 
                Rating Similarity: <span style="color: #FFD700; font-weight: 700;">{row.get('rating_similarity', 0)*100:.1f}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== MAIN APP ==========
def main():
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1>🤖 AI-Powered Electronics Recommender</h1>
        <p>Advanced Machine Learning | Hybrid Filtering | Deep Analytics</p>
        <div class="hero-badges">
            <span class="badge">🔬 TF-IDF + SVD</span>
            <span class="badge">🎯 Hybrid Algorithm</span>
            <span class="badge">📊 K-Means Clustering</span>
            <span class="badge">⚡ Real-time Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem; color: #00bfff;'>⚙️ Control Panel</h1>", unsafe_allow_html=True)
        
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=['xlsx', 'xls', 'csv'],
            help="Supports Excel and CSV formats"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Load & Process", use_container_width=True):
                with st.spinner("🔄 Processing data with advanced algorithms..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    if 'review_date' in df.columns:
                        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
                    
                    recommender = AdvancedElectronicsRecommender(df)
                    
                    algorithm = recommender.build_content_based_model(use_svd=True, n_components=100)
                    recommender.build_collaborative_features()
                    recommender.perform_clustering(n_clusters=5)
                    
                    st.session_state.df = df
                    st.session_state.recommender = recommender
                    st.session_state.data_loaded = True
                    st.session_state.algorithm_used = algorithm
                    
                    st.success("✅ Data processed successfully!")
                    st.rerun()
        
        if st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 🎛️ Algorithm Settings")
            
            content_weight = st.slider(
                "Content-Based Weight",
                0.0, 1.0, 0.7, 0.1,
                help="Weight for content similarity"
            )
            
            top_n = st.slider("Recommendations", 5, 20, 10)
            
            st.markdown("---")
            st.markdown("### 📊 Dataset Stats")
            df = st.session_state.df
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Products", f"{df['product_id'].nunique():,}")
                st.metric("Reviews", f"{len(df):,}")
            with col2:
                st.metric("Avg Rating", f"{df['star_rating'].mean():.2f}⭐")
                positive_pct = (df['star_rating'] >= 4).sum() / len(df) * 100
                st.metric("Positive", f"{positive_pct:.1f}%")
            
            if st.session_state.algorithm_used:
                st.markdown("---")
                st.markdown("### 🔬 Algorithm")
                st.markdown(f"""
                <div class="stats-box" style="text-align: center;">
                    <span class="algorithm-badge">{st.session_state.algorithm_used}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Main content
    if not st.session_state.data_loaded:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="info-box">
                <h2 style="text-align: center; color: #00bfff; margin-bottom: 1rem;">👋 Welcome to Advanced AI Recommendations!</h2>
                <p style="text-align: center; font-size: 1.1rem; color: #e0e0e0;">
                    Upload your electronics review dataset to experience cutting-edge AI analysis and recommendations.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("### 🌟 Advanced Features")
            
            features = {
                "🧠 Hybrid Algorithm": "Combines content-based and collaborative filtering with machine learning",
                "📊 TF-IDF + SVD": "Advanced text processing with latent semantic analysis",
                "🎯 K-Means Clustering": "Automatic product grouping and pattern detection",
                "⚡ Real-time Processing": "Fast similarity search with optimized algorithms",
                "🎨 Modern Dark Theme": "Beautiful blue-black interface with yellow accents",
                "📈 Deep Analytics": "15+ advanced visualization techniques",
                "🔗 Correlation Analysis": "Feature relationship exploration",
                "🌐 3D Visualization": "Multi-dimensional data exploration",
                "📦 Box & Violin Plots": "Statistical distribution analysis",
                "☀️ Sunburst Charts": "Hierarchical data visualization",
                "💾 Export Options": "Download in CSV and Excel formats"
            }
            
            for feature, desc in features.items():
                st.markdown(f"""
                <div class="feature-card">
                    <b>{feature}</b><br>
                    <span style="color: #b0b0b0; font-size: 0.9rem;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        df = st.session_state.df
        recommender = st.session_state.recommender
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Dashboard",
            "🔍 AI Recommendations",
            "📊 Advanced Analytics",
            "🎨 Visual Analytics",
            "📋 Data Explorer"
        ])
        
        # TAB 1: DASHBOARD
        with tab1:
            st.markdown("## 📊 Performance Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_products = df['product_id'].nunique()
                st.markdown(f"""
                <div class="metric-card" style="--gradient-start: #00bfff; --gradient-end: #0080ff;">
                    <div class="metric-icon">📦</div>
                    <div class="metric-label">Total Products</div>
                    <div class="metric-value">{total_products:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_reviews = len(df)
                st.markdown(f"""
                <div class="metric-card" style="--gradient-start: #ffd700; --gradient-end: #ffed4e;">
                    <div class="metric-icon">📝</div>
                    <div class="metric-label">Total Reviews</div>
                    <div class="metric-value">{total_reviews:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_rating = df['star_rating'].mean()
                st.markdown(f"""
                <div class="metric-card" style="--gradient-start: #00bfff; --gradient-end: #00d4ff;">
                    <div class="metric-icon">⭐</div>
                    <div class="metric-label">Average Rating</div>
                    <div class="metric-value">{avg_rating:.2f}</div>
                    <div class="metric-change positive">+{(avg_rating/5*100):.1f}% score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                positive_pct = (df['star_rating'] >= 4).sum() / len(df) * 100
                st.markdown(f"""
                <div class="metric-card" style="--gradient-start: #ffd700; --gradient-end: #ffa500;">
                    <div class="metric-icon">😊</div>
                    <div class="metric-label">Satisfaction Rate</div>
                    <div class="metric-value">{positive_pct:.1f}%</div>
                    <div class="metric-change positive">↑ Positive reviews</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_advanced_rating_chart(df)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_sentiment_donut(df)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Temporal analysis
            fig_time = create_time_series_chart(df)
            if fig_time:
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Clustering
            fig_cluster = create_cluster_visualization(df)
            if fig_cluster:
                st.plotly_chart(fig_cluster, use_container_width=True)
        
        # TAB 2: AI RECOMMENDATIONS
        with tab2:
            st.markdown("## 🔍 AI-Powered Product Recommendations")
            
            st.markdown('<div class="search-container">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_method = st.radio(
                    "Search Method:",
                    ["🏷️ By Product Title", "🔢 By Product ID", "🔎 Search Product Title"],
                    horizontal=True
                )
            
            with col2:
                top_n = st.number_input("Results", 5, 20, 10, key="search_top_n")
            
            with st.expander("⚙️ Advanced Settings", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    content_weight = st.slider(
                        "Content-Based Weight",
                        0.0, 1.0, 0.7, 0.05,
                        help="Weight for text similarity"
                    )
                with col2:
                    rating_weight = 1.0 - content_weight
                    st.metric("Rating Weight", f"{rating_weight:.2f}")
                
                st.info(f"🎯 **Hybrid Algorithm**: {content_weight*100:.0f}% Content + {rating_weight*100:.0f}% Rating")
            
            # Search by Product Title with live search
            if search_method == "🔎 Search Product Title":
                search_query = st.text_input(
                    "🔍 Search Product Title:",
                    placeholder="Type to search product titles...",
                    help="Enter product name to search"
                )
                
                if search_query:
                    # Filter products based on search query
                    matching_products = df[df['product_title'].str.contains(search_query, case=False, na=False)]['product_title'].unique()
                    
                    if len(matching_products) > 0:
                        st.success(f"✅ Found {len(matching_products)} matching products")
                        
                        selected_title = st.selectbox(
                            "Select from search results:",
                            options=matching_products,
                            help="Choose a product from search results"
                        )
                        
                        if st.button("🚀 Generate AI Recommendations", use_container_width=True):
                            with st.spinner("🤖 AI is analyzing products..."):
                                recommendations = recommender.get_hybrid_recommendations(
                                    product_title=selected_title,
                                    top_n=top_n,
                                    content_weight=content_weight,
                                    rating_weight=rating_weight
                                )
                                st.session_state.recommendations = recommendations
                    else:
                        st.warning(f"⚠️ No products found matching '{search_query}'. Try different keywords.")
                else:
                    st.info("💡 Start typing to search for products")
            
            elif search_method == "🏷️ By Product Title":
                product_titles = sorted(df['product_title'].unique())
                selected_title = st.selectbox(
                    "Select Product:",
                    options=product_titles,
                    help="Choose a product to find similar items"
                )
                
                if st.button("🚀 Generate AI Recommendations", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing products..."):
                        recommendations = recommender.get_hybrid_recommendations(
                            product_title=selected_title,
                            top_n=top_n,
                            content_weight=content_weight,
                            rating_weight=rating_weight
                        )
                        st.session_state.recommendations = recommendations
            else:
                product_ids = sorted(df['product_id'].unique())
                selected_id = st.selectbox(
                    "Select Product ID:",
                    options=product_ids
                )
                
                if st.button("🚀 Generate AI Recommendations", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing products..."):
                        recommendations = recommender.get_hybrid_recommendations(
                            product_id=selected_id,
                            top_n=top_n,
                            content_weight=content_weight,
                            rating_weight=rating_weight
                        )
                        st.session_state.recommendations = recommendations
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.session_state.recommendations is not None:
                recommendations = st.session_state.recommendations
                
                st.markdown("---")
                st.markdown("## ✨ AI Recommendations")
                
                st.markdown("""
                <div class="info-box">
                    <h4>🎯 Source Product</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if search_method in ["🏷️ By Product Title", "🔎 Search Product Title"]:
                    source = df[df['product_title'] == selected_title].iloc[0]
                else:
                    source = df[df['product_id'] == selected_id].iloc[0]
                
                st.markdown(f"""
                <div class="product-card" style="border-left: 5px solid #FFD700;">
                    <div class="product-title">🎯 {source['product_title']}</div>
                    <div class="product-rating">{"⭐" * int(source['star_rating'])} {source['star_rating']:.1f}/5.0</div>
                    <div class="product-meta">
                        <div class="meta-item">
                            <span>📝</span>
                            <span><b>{source.get('review_count', 'N/A')}</b> reviews</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f"### 🎁 Top {len(recommendations)} Similar Products")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_sim = recommendations['similarity_score'].mean()
                    st.metric("Avg Match Score", f"{avg_sim*100:.1f}%")
                with col2:
                    avg_rating = recommendations['star_rating'].mean()
                    st.metric("Avg Rating", f"{avg_rating:.2f}⭐")
                with col3:
                    total_reviews = recommendations['review_count'].sum()
                    st.metric("Total Reviews", f"{total_reviews:,.0f}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                for idx, row in recommendations.iterrows():
                    display_enhanced_product_card(row, int(row['rank']))
                
                st.markdown("---")
                st.markdown("### 💾 Export Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = recommendations.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name="ai_recommendations.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    try:
                        import io
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine="openpyxl") as writer:
                            recommendations.to_excel(writer, index=False, sheet_name="AI_Recommendations")
                        st.download_button(
                            label="📥 Download as Excel",
                            data=output.getvalue(),
                            file_name="ai_recommendations.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except:
                        pass
        
        # TAB 3: ADVANCED ANALYTICS
        with tab3:
            st.markdown("## 📊 Advanced Statistical Analysis")
            
            # Sentiment analysis
            sentiment_data = recommender.sentiment_analysis_advanced()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="stats-box">
                    <h4 style="color: #00bfff;">Mean Rating</h4>
                    <p style="font-size: 1.8rem; color: #ffd700; font-weight: 700;">{sentiment_data['mean_rating']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-box">
                    <h4 style="color: #00bfff;">Std Deviation</h4>
                    <p style="font-size: 1.8rem; color: #ffd700; font-weight: 700;">{sentiment_data['std_rating']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stats-box">
                    <h4 style="color: #00bfff;">Median Rating</h4>
                    <p style="font-size: 1.8rem; color: #ffd700; font-weight: 700;">{sentiment_data['median_rating']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stats-box">
                    <h4 style="color: #00bfff;">Skewness</h4>
                    <p style="font-size: 1.8rem; color: #ffd700; font-weight: 700;">{sentiment_data['skewness']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Correlation heatmap
            corr_matrix = recommender.correlation_analysis()
            if corr_matrix is not None:
                fig_corr = create_correlation_heatmap(corr_matrix)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Box plot
            col1, col2 = st.columns(2)
            with col1:
                fig_box = create_box_plot(df)
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                fig_violin = create_violin_plot(df)
                st.plotly_chart(fig_violin, use_container_width=True)
        
        # TAB 4: VISUAL ANALYTICS
        with tab4:
            st.markdown("## 🎨 Advanced Visualizations")
            
            # Funnel chart
            fig_funnel = create_funnel_chart(df)
            st.plotly_chart(fig_funnel, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sunburst
                if 'verified_purchase' in df.columns:
                    fig_sun = create_sunburst_chart(df)
                    if fig_sun:
                        st.plotly_chart(fig_sun, use_container_width=True)
                    else:
                        st.info("☀️ Hierarchical chart tidak dapat ditampilkan. Data mungkin tidak memiliki kolom 'verified_purchase' atau data tidak cukup.")
                else:
                    st.warning("⚠️ Kolom 'verified_purchase' tidak ditemukan dalam dataset. Sunburst chart memerlukan kolom ini untuk analisis hierarki.")
            
            with col2:
                # Cluster visualization
                fig_cluster = create_cluster_visualization(df)
                if fig_cluster:
                    st.plotly_chart(fig_cluster, use_container_width=True)
                else:
                    st.info("🎯 Cluster visualization sedang diproses...")
            
            # 3D Scatter
            st.markdown("### 🌐 3D Feature Space")
            fig_3d = create_scatter_3d(df)
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Additional visualizations
            st.markdown("---")
            st.markdown("### 📊 Additional Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Product performance scatter
                st.markdown("#### 🎯 Product Performance Matrix")
                
                product_perf = df.groupby('product_id').agg({
                    'star_rating': 'mean',
                    'product_title': 'first',
                    'helpful_votes': 'sum',
                    'review_body': 'count'
                }).rename(columns={'review_body': 'review_count'})
                
                product_perf = product_perf[product_perf['review_count'] >= 3].sort_values('star_rating', ascending=False).head(50)
                
                fig_perf = go.Figure(data=[
                    go.Scatter(
                        x=product_perf['review_count'],
                        y=product_perf['star_rating'],
                        mode='markers',
                        marker=dict(
                            size=product_perf['helpful_votes']/5,
                            color=product_perf['star_rating'],
                            colorscale=[[0, '#E74C3C'], [0.5, '#ffd700'], [1, '#00bfff']],
                            showscale=True,
                            colorbar=dict(title='Rating'),
                            line=dict(width=1, color='#1a1a2e')
                        ),
                        text=product_perf['product_title'],
                        hovertemplate='<b>%{text}</b><br>Rating: %{y:.2f}⭐<br>Reviews: %{x}<extra></extra>'
                    )
                ])
                
                fig_perf.update_layout(
                    xaxis_title='Number of Reviews',
                    yaxis_title='Average Rating',
                    template='plotly_dark',
                    height=400,
                    plot_bgcolor='#1a1a2e',
                    paper_bgcolor='#1a1a2e',
                    font=dict(color='#e0e0e0')
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
            
            with col2:
                # Rating vs Helpfulness
                st.markdown("#### 📈 Rating vs Helpfulness")
                
                help_by_rating = df[df['total_votes'] > 0].groupby('star_rating').agg({
                    'helpful_votes': 'sum',
                    'total_votes': 'sum'
                }).reset_index()
                
                help_by_rating['helpful_ratio'] = help_by_rating['helpful_votes'] / help_by_rating['total_votes']
                
                fig_help = go.Figure()
                
                fig_help.add_trace(go.Scatter(
                    x=help_by_rating['star_rating'],
                    y=help_by_rating['helpful_ratio'],
                    mode='lines+markers',
                    name='Helpful Ratio',
                    line=dict(color='#ffd700', width=4),
                    marker=dict(size=12, color='#ffd700'),
                    fill='tozeroy',
                    fillcolor='rgba(255, 215, 0, 0.2)'
                ))
                
                fig_help.update_layout(
                    xaxis_title='Star Rating',
                    yaxis_title='Helpful Ratio',
                    template='plotly_dark',
                    height=400,
                    plot_bgcolor='#1a1a2e',
                    paper_bgcolor='#1a1a2e',
                    font=dict(color='#e0e0e0')
                )
                
                st.plotly_chart(fig_help, use_container_width=True)
        
        # TAB 5: DATA EXPLORER
        with tab5:
            st.markdown("## 📋 Data Explorer & Filter")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rating_filter = st.multiselect(
                    "⭐ Rating",
                    options=[1, 2, 3, 4, 5],
                    default=[1, 2, 3, 4, 5]
                )
            
            with col2:
                if 'verified_purchase' in df.columns:
                    verified_filter = st.multiselect(
                        "✓ Verified",
                        options=df['verified_purchase'].unique(),
                        default=df['verified_purchase'].unique()
                    )
                else:
                    verified_filter = None
            
            with col3:
                min_helpful = st.number_input("Min Helpful Votes", 0, int(df['helpful_votes'].max()), 0)
            
            with col4:
                sample_size = st.number_input(
                    "Sample Size",
                    10, len(df), min(500, len(df))
                )
            
            # Apply filters
            filtered_df = df[df['star_rating'].isin(rating_filter)]
            if verified_filter and 'verified_purchase' in df.columns:
                filtered_df = filtered_df[filtered_df['verified_purchase'].isin(verified_filter)]
            filtered_df = filtered_df[filtered_df['helpful_votes'] >= min_helpful]
            
            # Statistics
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="stats-box">
                    <h4 style="color: #00bfff;">Filtered Rows</h4>
                    <p style="font-size: 1.5rem; color: #ffd700; font-weight: 700;">{len(filtered_df):,}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-box">
                    <h4 style="color: #00bfff;">Unique Products</h4>
                    <p style="font-size: 1.5rem; color: #ffd700; font-weight: 700;">{filtered_df['product_id'].nunique():,}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stats-box">
                    <h4 style="color: #00bfff;">Avg Rating</h4>
                    <p style="font-size: 1.5rem; color: #ffd700; font-weight: 700;">{filtered_df['star_rating'].mean():.2f}⭐</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                pct = len(filtered_df) / len(df) * 100
                st.markdown(f"""
                <div class="stats-box">
                    <h4 style="color: #00bfff;">% of Total</h4>
                    <p style="font-size: 1.5rem; color: #ffd700; font-weight: 700;">{pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display data
            display_cols = [
                'product_title', 'star_rating', 'review_headline',
                'helpful_votes', 'total_votes', 'verified_purchase',
                'word_count', 'text_length'
            ]
            display_cols = [col for col in display_cols if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[display_cols].head(sample_size),
                use_container_width=True,
                height=500
            )
            
            # Export
            st.markdown("---")
            st.markdown("### 💾 Export Filtered Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "filtered_data.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                try:
                    import io
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        filtered_df.to_excel(writer, index=False)
                    st.download_button(
                        "📥 Download Excel",
                        output.getvalue(),
                        "filtered_data.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except:
                    pass
            
            # Additional Statistics Section
            st.markdown("---")
            st.markdown("### 📈 Detailed Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-box">
                    <h4>📝 Text Statistics</h4>
                </div>
                """, unsafe_allow_html=True)
                
                text_stats = pd.DataFrame({
                    'Metric': [
                        'Average Text Length',
                        'Average Word Count',
                        'Max Text Length',
                        'Min Text Length',
                        'Median Word Count'
                    ],
                    'Value': [
                        f"{filtered_df['text_length'].mean():.1f} chars",
                        f"{filtered_df['word_count'].mean():.1f} words",
                        f"{filtered_df['text_length'].max():,} chars",
                        f"{filtered_df['text_length'].min():,} chars",
                        f"{filtered_df['word_count'].median():.0f} words"
                    ]
                })
                st.dataframe(text_stats, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("""
                <div class="info-box">
                    <h4>👍 Helpfulness Metrics</h4>
                </div>
                """, unsafe_allow_html=True)
                
                has_votes = filtered_df[filtered_df['total_votes'] > 0]
                if len(has_votes) > 0:
                    help_stats = pd.DataFrame({
                        'Metric': [
                            'Reviews with Votes',
                            'Average Helpful Ratio',
                            'Average Helpful Votes',
                            'Average Total Votes',
                            'Max Helpful Votes'
                        ],
                        'Value': [
                            f"{len(has_votes):,} ({(len(has_votes)/len(filtered_df))*100:.1f}%)",
                            f"{has_votes['helpful_ratio'].mean():.3f}",
                            f"{has_votes['helpful_votes'].mean():.1f}",
                            f"{has_votes['total_votes'].mean():.1f}",
                            f"{has_votes['helpful_votes'].max():.0f}"
                        ]
                    })
                    st.dataframe(help_stats, use_container_width=True, hide_index=True)
                else:
                    st.info("No reviews with votes in filtered data")
            
            # Distribution charts for filtered data
            st.markdown("---")
            st.markdown("### 📊 Filtered Data Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating distribution for filtered data
                fig_filtered_rating = go.Figure(data=[
                    go.Bar(
                        x=filtered_df['star_rating'].value_counts().sort_index().index,
                        y=filtered_df['star_rating'].value_counts().sort_index().values,
                        marker=dict(
                            color=['#E74C3C', '#E67E22', '#F39C12', '#ffd700', '#00bfff'],
                            line=dict(color='#1a1a2e', width=2)
                        ),
                        text=filtered_df['star_rating'].value_counts().sort_index().values,
                        textposition='outside',
                        textfont=dict(color='#e0e0e0')
                    )
                ])
                
                fig_filtered_rating.update_layout(
                    title={
                        'text': 'Filtered Rating Distribution',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'color': '#00bfff'}
                    },
                    xaxis_title='Star Rating',
                    yaxis_title='Count',
                    template='plotly_dark',
                    height=400,
                    showlegend=False,
                    plot_bgcolor='#1a1a2e',
                    paper_bgcolor='#1a1a2e',
                    font=dict(color='#e0e0e0')
                )
                
                st.plotly_chart(fig_filtered_rating, use_container_width=True)
            
            with col2:
                # Word count histogram for filtered data
                fig_filtered_words = go.Figure(data=[
                    go.Histogram(
                        x=filtered_df['word_count'],
                        nbinsx=30,
                        marker=dict(
                            color='#00bfff',
                            line=dict(color='#1a1a2e', width=1)
                        )
                    )
                ])
                
                fig_filtered_words.update_layout(
                    title={
                        'text': 'Filtered Word Count Distribution',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'color': '#00bfff'}
                    },
                    xaxis_title='Word Count',
                    yaxis_title='Frequency',
                    template='plotly_dark',
                    height=400,
                    showlegend=False,
                    plot_bgcolor='#1a1a2e',
                    paper_bgcolor='#1a1a2e',
                    font=dict(color='#e0e0e0')
                )
                
                st.plotly_chart(fig_filtered_words, use_container_width=True)

# ========== RUN APP ==========
if __name__ == "__main__":
    main()