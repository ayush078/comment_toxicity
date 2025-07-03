import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def build_and_train_model(X_train_vec, train_df):
    print("Building and training the Logistic Regression model...")

    # Define target columns
    target_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_train = train_df[target_columns].values

    # Train a separate Logistic Regression model for each toxicity type
    models = {}
    for i, col in enumerate(target_columns):
        print(f"Training model for: {col}")
        model = LogisticRegression(solver="liblinear", random_state=42, n_jobs=-1) # Using liblinear for multiclass/binary and n_jobs for parallel processing
        model.fit(X_train_vec, y_train[:, i])
        models[col] = model

    print("Model training complete.")
    joblib.dump(models, "toxicity_logistic_regression_models.pkl")
    print("Models saved to toxicity_logistic_regression_models.pkl")
    return models

if __name__ == "__main__":
    # Load preprocessed data and vectorizer
    train_df_processed = pd.read_pickle("train_processed.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    # Re-vectorize the processed text to get X_train_vec
    X_train_vec = tfidf_vectorizer.transform(train_df_processed["comment_text_processed"])

    models = build_and_train_model(X_train_vec, train_df_processed)


