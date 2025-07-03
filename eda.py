import pandas as pd
import io
import sys

def perform_eda(train_path, test_path):
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Datasets loaded successfully.")

    eda_output = []

    eda_output.append("--- Train Dataset Info ---")
    buffer = io.StringIO()
    sys.stdout = buffer
    train_df.info()
    sys.stdout = sys.__stdout__
    eda_output.append(buffer.getvalue())

    eda_output.append("\n--- Test Dataset Info ---")
    buffer = io.StringIO()
    sys.stdout = buffer
    test_df.info()
    sys.stdout = sys.__stdout__
    eda_output.append(buffer.getvalue())

    eda_output.append("\n--- Train Dataset Head ---")
    eda_output.append(train_df.head().to_string())
    eda_output.append("\n--- Test Dataset Head ---")
    eda_output.append(test_df.head().to_string())

    eda_output.append("\n--- Train Dataset Description ---")
    eda_output.append(train_df.describe(include='all').to_string())
    eda_output.append("\n--- Test Dataset Description ---")
    eda_output.append(test_df.describe(include='all').to_string())

    eda_output.append("\n--- Missing Values in Train Dataset ---")
    eda_output.append(train_df.isnull().sum().to_string())
    eda_output.append("\n--- Missing Values in Test Dataset ---")
    eda_output.append(test_df.isnull().sum().to_string())

    eda_output.append("\n--- Duplicate Values in Train Dataset ---")
    eda_output.append(str(train_df.duplicated().sum()))
    eda_output.append("\n--- Duplicate Values in Test Dataset ---")
    eda_output.append(str(test_df.duplicated().sum()))

    toxicity_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    for col in toxicity_columns:
        eda_output.append(f"\n--- Value Counts for \'{col}\' in Train Dataset ---")
        eda_output.append(train_df[col].value_counts().to_string())

    with open("eda_results.txt", "w") as f:
        f.write("\n".join(eda_output))
    print("EDA results saved to eda_results.txt")

if __name__ == "__main__":
    perform_eda("train.csv", "test.csv")


