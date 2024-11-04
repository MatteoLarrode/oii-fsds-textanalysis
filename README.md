# Text Analysis of Martial Arts-related Subreddits

The purpose of this project is to analyse the differences and similarities between different subreddits existing around a common topic, in this case martial arts. At first, I will collect Reddit data, analyse TF-IDF scores, and attempt to classify the subreddits using k-means and Naive Bayes algorithms. Then, I will introduce networks to visualise connections between threaded comments and users. Finally, I will make use of embeddings to further the analysis.

## I/ Collecting Reddit Data and Exploring TF-IDF Results


```python
# Setup autoreload
%load_ext autoreload
%autoreload 2

# Create README.md 
# pip3 install nbconvert
# jupyter nbconvert --execute --to markdown MartialArtsRedditAnalysis.ipynb
# then rename to README.md
```


```python
from reddit_helper import *
import os
import pyarrow
import pickle
```

Let's first collect  50 posts from the r/MuayThai, r/bjj, r/MMA, and r/Boxing subreddits, and display some of their initial characteristics. This is done using the `RedditScraper` class created in the `reddit_helper.py` file. 


```python
# Example subreddits
subreddits = ['MuayThai', 'bjj', 'MMA', 'Boxing']

# Establish cache directory
CACHE_DIR = os.path.join(os.getcwd(), 'data')

# Analysis parameters
MAX_TERMS = 1000
MIN_DOC_FREQ = 2
TOP_N = 10
# LIMIT = 50 Initial limit at 50 for testing
LIMIT = 500
USERNAME = "matteolarrode"

# Initialize scraper
scraper = RedditScraper(
user_agent=f"SDS_textanalysis/1.0 (by /u/{USERNAME})"
)

# Analyze each subreddit independently
results = {}
submissions = {}

for subreddit in subreddits:
    print(f"\nAnalyzing r/{subreddit}...")

    # Define the cache file path for this subreddit
    cached_file = os.path.join(CACHE_DIR, f"{subreddit}_data.pkl")

    # If the data for the subreddit is already cached, no need to collect
    if os.path.exists(cached_file):
        # Load data from cache
        print(f"Loading cached data for r/{subreddit}...")
        with open(cached_file, 'rb') as file:
            submissions[subreddit] = pickle.load(file)
    
    # Otherwise, collect posts and cache them
    else:
        submissions[subreddit] = scraper.get_subreddit_posts(subreddit, limit=LIMIT)
        with open(cached_file, 'wb') as file:
            pickle.dump(submissions[subreddit], file)

    # Analyze subreddit
    results[subreddit] = analyze_subreddit(
        submissions[subreddit],
        max_terms=MAX_TERMS,   # Maximum number of terms to keep
        min_doc_freq=MIN_DOC_FREQ,   # Term must appear in at least min_doc_freq documents
        top_n_terms=TOP_N # Number of top terms returned in result
    )

    # Print results for this subreddit
    print(f"\nVocabulary Statistics for r/{subreddit}:")
    print(f"Total words: {results[subreddit]['vocab_stats']['total_words']}")
    print(f"Unique words: {results[subreddit]['vocab_stats']['unique_words']}")
    print(f"Words appearing ≥{MIN_DOC_FREQ} times: {results[subreddit]['vocab_stats']['words_min_freq']}")
    print(f"Coverage by top {MAX_TERMS} words: {results[subreddit]['vocab_stats']['coverage_top_1000']:.2f}%")
    print(f"Matrix shape: {results[subreddit]['matrix_shape']}")
    print(f"Matrix sparsity: {results[subreddit]['matrix_sparsity']:.2f}%")

    print(f"\nTop {TOP_N} terms by TF-IDF score:")
    print(results[subreddit]['top_terms'][['term', 'score']].to_string())
```

    
    Analyzing r/MuayThai...
    Loading cached data for r/MuayThai...
    
    Vocabulary Statistics for r/MuayThai:
    Total words: 38142
    Unique words: 5155
    Words appearing ≥2 times: 2384
    Coverage by top 1000 words: 82.93%
    Matrix shape: (500, 1000)
    Matrix sparsity: 97.76%
    
    Top 10 terms by TF-IDF score:
             term     score
    860      thai  0.051784
    527      muay  0.050555
    266     fight  0.042132
    388        im  0.035211
    932        vs  0.030987
    560       one  0.029892
    907  training  0.025881
    454      like  0.025353
    776  sparring  0.022809
    307       get  0.022165
    
    Analyzing r/bjj...
    Loading cached data for r/bjj...
    
    Vocabulary Statistics for r/bjj:
    Total words: 46080
    Unique words: 5572
    Words appearing ≥2 times: 2717
    Coverage by top 1000 words: 83.23%
    Matrix shape: (500, 1000)
    Matrix sparsity: 97.37%
    
    Top 10 terms by TF-IDF score:
           term     score
    91      bjj  0.045379
    387      im  0.037150
    465    like  0.033840
    306      gi  0.029001
    331   guard  0.027802
    303     get  0.027212
    337     gym  0.026797
    83     belt  0.026350
    45   anyone  0.022571
    318    good  0.022170
    
    Analyzing r/MMA...
    Loading cached data for r/MMA...
    
    Vocabulary Statistics for r/MMA:
    Total words: 22560
    Unique words: 4140
    Words appearing ≥2 times: 1872
    Coverage by top 1000 words: 81.10%
    Matrix shape: (500, 1000)
    Matrix sparsity: 98.31%
    
    Top 10 terms by TF-IDF score:
            term     score
    945       vs  0.068714
    916      ufc  0.062744
    334    fight  0.048341
    825  spoiler  0.038149
    50       308  0.027527
    567      mma  0.024670
    512     link  0.023135
    630  oktagon  0.022059
    897    title  0.019959
    345    flair  0.018679
    
    Analyzing r/Boxing...
    Loading cached data for r/Boxing...
    
    Vocabulary Statistics for r/Boxing:
    Total words: 34765
    Unique words: 5074
    Words appearing ≥2 times: 2429
    Coverage by top 1000 words: 81.35%
    Matrix shape: (500, 1000)
    Matrix sparsity: 97.93%
    
    Top 10 terms by TF-IDF score:
              term     score
    932         vs  0.046524
    314      fight  0.045823
    143     boxing  0.035954
    132      bivol  0.032466
    122  beterbiev  0.031438
    68         amp  0.020896
    88       artur  0.015270
    869      think  0.015031
    883      title  0.014548
    624        one  0.014423


Let's try to grasp a better understanding of the data structure of the `results` object. It is a dictionary with the following keys:



```python
print(results['MuayThai'].keys())
print(results['MuayThai']['vocab_stats'].keys())
```

    dict_keys(['vocab_stats', 'freq_distribution', 'top_terms', 'vectorizer', 'matrix_shape', 'matrix_sparsity'])
    dict_keys(['total_words', 'unique_words', 'words_min_freq', 'coverage_top_1000'])


- `vocab_stats` is itself a dictionary which contains some statistics on the vocabulary present in the posts of the subreddit.
- `top_terms` returns a number of the top terms by TF-IDF defined by the `TOP_N` constant.

Also, the code has been written so that the first time it is run to query the posts of a subreddit, the data is cached in `/data` for later use as a `pickle` file (`/data` is in the .gitignore). Note that `.pkl` files can only be read in Python.

### Some exploratory data analysis

First, I will **plot keywords over time**. Each post is a dictionary, and its creation Unix timestamp is associated to te `created_utc` key. 


```python
print(f"Keys of each post: {submissions['MuayThai'][0].keys()}")
```

    Keys of each post: dict_keys(['approved_at_utc', 'subreddit', 'selftext', 'author_fullname', 'saved', 'mod_reason_title', 'gilded', 'clicked', 'title', 'link_flair_richtext', 'subreddit_name_prefixed', 'hidden', 'pwls', 'link_flair_css_class', 'downs', 'thumbnail_height', 'top_awarded_type', 'hide_score', 'name', 'quarantine', 'link_flair_text_color', 'upvote_ratio', 'author_flair_background_color', 'subreddit_type', 'ups', 'total_awards_received', 'media_embed', 'thumbnail_width', 'author_flair_template_id', 'is_original_content', 'user_reports', 'secure_media', 'is_reddit_media_domain', 'is_meta', 'category', 'secure_media_embed', 'link_flair_text', 'can_mod_post', 'score', 'approved_by', 'is_created_from_ads_ui', 'author_premium', 'thumbnail', 'edited', 'author_flair_css_class', 'author_flair_richtext', 'gildings', 'content_categories', 'is_self', 'mod_note', 'created', 'link_flair_type', 'wls', 'removed_by_category', 'banned_by', 'author_flair_type', 'domain', 'allow_live_comments', 'selftext_html', 'likes', 'suggested_sort', 'banned_at_utc', 'view_count', 'archived', 'no_follow', 'is_crosspostable', 'pinned', 'over_18', 'all_awardings', 'awarders', 'media_only', 'can_gild', 'spoiler', 'locked', 'author_flair_text', 'treatment_tags', 'visited', 'removed_by', 'num_reports', 'distinguished', 'subreddit_id', 'author_is_blocked', 'mod_reason_by', 'removal_reason', 'link_flair_background_color', 'id', 'is_robot_indexable', 'report_reasons', 'author', 'discussion_type', 'num_comments', 'send_replies', 'contest_mode', 'mod_reports', 'author_patreon_flair', 'author_flair_text_color', 'permalink', 'stickied', 'url', 'subreddit_subscribers', 'created_utc', 'num_crossposts', 'media', 'is_video'])



```python
print(submissions['MuayThai'][0]['selftext'])
print(submissions['MuayThai'][0]['created_utc'])
```

    Hey guys. I need to cut 2.5-3kg (5.5lbs) in 10 days. Im running everyday, im on caloric deficit and im gonna cut water in last 2/3 days. Any other tips? I've heard about keto and throwing out carbs.
    1730323742.0


Using helper functions created in `reddit_helper.py`, I create a `DataFrame` with each post, their text, creation date and the counts and term-frequencies of their top-terms.


```python
# Top 10 words (TF-IDF)
top_terms = results['MuayThai']['top_terms']['term'].tolist()

# Processing the result of the Reddit scraping for r/MuayThai
# Creates a DataFrame with the counts and TF of top terms
word_freq_muaythai_df = process_posts(submissions['MuayThai'], 'MuayThai', top_terms)

# Check the structure of the DataFrame
word_freq_muaythai_df.columns
```




    Index(['post_id', 'created_utc_unix', 'created_date', 'thai_count', 'thai_tf',
           'muay_count', 'muay_tf', 'fight_count', 'fight_tf', 'im_count', 'im_tf',
           'vs_count', 'vs_tf', 'one_count', 'one_tf', 'training_count',
           'training_tf', 'like_count', 'like_tf', 'sparring_count', 'sparring_tf',
           'get_count', 'get_tf', 'total_word_count'],
          dtype='object')




```python
# Get top 5 terms instead of 10 for better clarity
top5_terms = results['MuayThai']['top_terms']['term'].head(5).tolist()

plot_top_words_freq(word_freq_muaythai_df, top5_terms, title='Daily Average TF of Top 5 Words in Muay Thai Subreddit')
```


    
![png](MartialArtsRedditAnalysis_files/MartialArtsRedditAnalysis_15_0.png)
    


This plot shows the daily average term frequency (TF) of the top five words from the Muay Thai subreddit ("thai," "muay," "fight," "im," and "vs") over a period of about a month. Superficially, fluctuations in the prominence of these words could reflect the ebb and flow of discussions surrounding specific topics or events related to Muay Thai, such as training discussions or high-profile fights. 

Perhaps the most telling example is the peak of the word 'fight' on October 7th 2024, which corresponds to the RWS Muay Thai tournament that happened on that day, and was at the center of discussions on the Reddit.
