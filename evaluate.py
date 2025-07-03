import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_and_predict(test_path, train_df_processed, tfidf_vectorizer, models):
    print("Loading test dataset for prediction...")
    test_df = pd.read_csv(test_path)
    print("Test dataset loaded.")

    print("Preprocessing test data...")
    test_df_processed = pd.read_pickle("test_processed.pkl")
    X_test_vec = tfidf_vectorizer.transform(test_df_processed["comment_text_processed"])
    print("Test data preprocessed and vectorized.")

    target_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    predictions = pd.DataFrame({"id": test_df["id"]})

    print("Generating predictions on test set...")
    for col in target_columns:
        model = models[col]
        predictions[col] = model.predict(X_test_vec)
    print("Predictions generated.")

    predictions.to_csv("test_predictions.csv", index=False)
    print("Test predictions saved to test_predictions.csv")

    print("\nEvaluating models on training data (for performance metrics)...")
    X_train_vec = tfidf_vectorizer.transform(train_df_processed["comment_text_processed"])
    y_train = train_df_processed[target_columns].values

    evaluation_results = {}
    for i, col in enumerate(target_columns):
        model = models[col]
        y_pred = model.predict(X_train_vec)
        y_true = y_train[:, i]

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)

        evaluation_results[col] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc_score": roc_auc
        }
        print(f"--- {col} Model Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")

    with open("evaluation_results.txt", "w") as f:
        for col, metrics in evaluation_results.items():
            f.write(f"--- {col} Model Evaluation ---\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
            f.write(f"ROC AUC Score: {metrics['roc_auc_score']:.4f}\n\n")
    print("Evaluation results saved to evaluation_results.txt")

if __name__ == "__main__":
    # Load preprocessed data, vectorizer, and models
    train_df_processed = pd.read_pickle("train_processed.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    models = joblib.load("toxicity_logistic_regression_models.pkl")

    evaluate_and_predict("test.csv", train_df_processed, tfidf_vectorizer, models)


