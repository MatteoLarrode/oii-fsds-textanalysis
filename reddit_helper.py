import requests
import time
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re

class RedditScraper:
    def __init__(self, user_agent):
        """
        Initialize the scraper with a user agent string.
        Example user agent: "SDS_textanalysis/1.0 (by /u/your_username)"
        """
        self.headers = {'User-Agent': user_agent}
        self.base_url = "https://api.reddit.com"
        
    def get_subreddit_posts(self, subreddit, limit=100):
        """
        Collect posts from a subreddit with proper pagination and rate limiting.
        """
        posts = []
        after = None
        
        while len(posts) < limit:
            url = f"{self.base_url}/r/{subreddit}/new"
            params = {
                'limit': min(100, limit - len(posts)),
                'after': after
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"Error accessing r/{subreddit}: {response.status_code}")
                break
                
            data = response.json()
            new_posts = data['data']['children']
            if not new_posts:
                break
                
            posts.extend([post['data'] for post in new_posts])
            after = data['data']['after']
            
            if not after:
                break
                
            time.sleep(2)
            
        return posts[:limit]

def preprocess_text(text):
    """
    Clean and normalize text.
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_vocabulary(texts, min_freq=2):
    """
    Analyze vocabulary distribution in a corpus.
    Returns word frequencies and vocabulary statistics.
    """
    # Tokenize all texts
    words = ' '.join(texts).split()
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Calculate vocabulary statistics
    total_words = len(words)
    unique_words = len(word_freq)
    
    # Create frequency distribution DataFrame
    freq_df = pd.DataFrame(list(word_freq.items()), columns=['word', 'frequency'])
    freq_df['percentage'] = freq_df['frequency'] / total_words * 100
    freq_df = freq_df.sort_values('frequency', ascending=False)
    
    # Calculate cumulative coverage
    freq_df['cumulative_percentage'] = freq_df['percentage'].cumsum()
    
    stats = {
        'total_words': total_words,
        'unique_words': unique_words,
        'words_min_freq': sum(1 for freq in word_freq.values() if freq >= min_freq),
        'coverage_top_1000': freq_df.iloc[:1000]['frequency'].sum() / total_words * 100 if len(freq_df) >= 1000 else 100
    }
    
    return freq_df, stats

def analyze_subreddit(posts, max_terms=1000, min_doc_freq=2):
    """
    Analyze a single subreddit's posts independently.
    """
    # Combine title and selftext
    texts = [
        preprocess_text(post.get('title', '')) + ' ' + 
        preprocess_text(post.get('selftext', ''))
        for post in posts
    ]
    
    # Analyze vocabulary first
    freq_df, vocab_stats = analyze_vocabulary(texts, min_freq=min_doc_freq)
    
    # Initialize TF-IDF vectorizer for this subreddit
    stop_words = list(set(stopwords.words('english')))
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_terms,
        min_df=min_doc_freq
    )
    
    # Compute TF-IDF
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Get average TF-IDF scores
    mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
    
    # Get top terms
    feature_names = vectorizer.get_feature_names_out()
    top_terms = pd.DataFrame({
        'term': feature_names,
        'score': mean_tfidf
    }).sort_values('score', ascending=False)
    
    return {
        'vocab_stats': vocab_stats,
        'freq_distribution': freq_df,
        'top_terms': top_terms.head(),
        'vectorizer': vectorizer,
        'matrix_shape': tfidf_matrix.shape,
        'matrix_sparsity': 100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
    }