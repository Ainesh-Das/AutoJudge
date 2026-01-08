import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HIGH_DIFFICULTY_WORDS = [
    "graph", "tree", "node", "edge", "lowest common ancestor",
     "minimum spanning tree","flow", "max flow"
]

MEDIUM_DIFFICULTY_WORDS = [
     "bitmask", "recursively", "shortest path",
    "maximum of minimum", "minimum of maximum"
]

def count_keywords(text, keywords):
    text = text.lower()
    count = 0
    for word in keywords:
        escaped_word = re.escape(word)
        count += len(re.findall(r'\b' + escaped_word + r'\b', text))
    return count

def get_manual_features(text):
    text = text.lower()
    text_length = len(text)
    math_symbols = len(re.findall(r'[+^\-*/|&=<>!%]', text))
    high_diff_score = count_keywords(text, HIGH_DIFFICULTY_WORDS)
    medium_diff_score = count_keywords(text, MEDIUM_DIFFICULTY_WORDS)
    return [text_length, math_symbols, high_diff_score, medium_diff_score]

def add_manual_features_df(df):
    features_list = df['combined_text'].apply(get_manual_features).tolist()
    features_df = pd.DataFrame(features_list, columns=['text_length', 'math_symbols', 'high_diff_score', 'medium_diff_score'])
    df = pd.concat([df, features_df], axis=1)
    return df

def extract_features_for_training(df):
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2) 
    )

    X_text = tfidf.fit_transform(df['combined_text'])
    X_manual = df[
        ['text_length', 'math_symbols', 'high_diff_score', 'medium_diff_score']
    ].values

    return X_text, X_manual, tfidf