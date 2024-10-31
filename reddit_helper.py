import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
from datetime import datetime, timezone
from dateutil import tz

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

def analyze_subreddit(posts, max_terms=1000, min_doc_freq=2, top_n_terms=5):
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
        'top_terms': top_terms.head(top_n_terms),
        'vectorizer': vectorizer,
        'matrix_shape': tfidf_matrix.shape,
        'matrix_sparsity': 100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
    }

def preprocess_text(text):
    """
    Remove punctuation and normalize text to lowercase.
    """
    clean_text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return clean_text.lower()

def calculate_term_frequency(text, terms):
    """
    Calculate term frequency for each specified term in the given text.
    
    Returns:
    - A dictionary with term counts and term frequencies.
    """
    term_counts = {}
    words = text.split()
    total_word_count = len(words)
    
    for term in terms:
        term_count = words.count(term.lower())
        term_tf = term_count / total_word_count if total_word_count > 0 else 0.0
        term_counts[f'{term}_count'] = term_count
        term_counts[f'{term}_tf'] = term_tf
    
    return term_counts, total_word_count

def process_posts(submissions, subreddit_name, top_terms):
    """
    Process submissions to extract post information, term frequencies, and other relevant metrics.
    
    Parameters:
    - submissions: List of post dictionaries for a subreddit
    - subreddit_name: Name of the subreddit being processed (for display/clarity purposes)
    - top_terms: List of terms for which to calculate term frequencies
    
    Returns:
    - DataFrame with post data, including term counts, frequencies, and metadata.
    """
    data = []
    
    for i, post in enumerate(submissions):
        # Initialize post data dictionary
        post_data = {
            'post_id': post.get('id', i),  # Use post 'id' if available; otherwise, fallback to index
            'created_utc_unix': post['created_utc'],
            'created_date': datetime.fromtimestamp(post['created_utc'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Preprocess text and calculate term frequencies
        text = preprocess_text(post.get('selftext', ''))
        term_counts, total_word_count = calculate_term_frequency(text, top_terms)
        
        # Update post_data with term counts and total word count
        post_data.update(term_counts)
        post_data['total_word_count'] = total_word_count
        
        data.append(post_data)
    
    # Create a DataFrame from the list of dictionaries
    return pd.DataFrame(data)