import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib

# Set NLTK data path and download necessary resources
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt", download_dir=nltk_data_path)

try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords", download_dir=nltk_data_path)


def preprocess_text_series(series):
    series = series.str.lower()
    series = series.apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    return series

def tokenize_and_remove_stopwords_series(series):
    stop_words = set(stopwords.words('english'))
    return series.apply(lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words]))

def perform_preprocessing(train_path, test_path):
    print("Loading datasets for preprocessing...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Datasets loaded.")

    print("Applying text preprocessing...")
    train_df["comment_text_processed"] = preprocess_text_series(train_df["comment_text"])
    test_df["comment_text_processed"] = preprocess_text_series(test_df["comment_text"])
    print("Text preprocessing complete.")

    print("Applying tokenization and stopword removal...")
    train_df["comment_text_processed"] = tokenize_and_remove_stopwords_series(train_df["comment_text_processed"])
    test_df["comment_text_processed"] = tokenize_and_remove_stopwords_series(test_df["comment_text_processed"])
    print("Tokenization and stopword removal complete.")

    print("Performing TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=10000) # Limiting features for demonstration
    X_train = vectorizer.fit_transform(train_df["comment_text_processed"])
    X_test = vectorizer.transform(test_df["comment_text_processed"])
    print("TF-IDF vectorization complete.")

    print("Saving processed dataframes and vectorizer...")
    train_df.to_pickle("train_processed.pkl")
    test_df.to_pickle("test_processed.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    
    print("Preprocessing and feature engineering complete. Processed data saved to train_processed.pkl, test_processed.pkl, and tfidf_vectorizer.pkl.")
    return X_train, X_test, train_df, test_df

if __name__ == "__main__":
    X_train_vec, X_test_vec, train_df_processed, test_df_processed = perform_preprocessing("train.csv", "test.csv")
    print(f"Shape of X_train_vec: {X_train_vec.shape}")
    print(f"Shape of X_test_vec: {X_test_vec.shape}")
    print("First 5 processed comments from train_df:")
    print(train_df_processed["comment_text_processed"].head().to_string())


