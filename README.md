# Comment Toxicity Classification

## Project Overview
This project aims to develop a machine learning model to classify online comments based on their toxicity. The goal is to identify and flag toxic comments, contributing to healthier online communication environments. This README provides an overview of the project, instructions for setting up and running the code, and details about the model and its performance.

## Setup and Installation
To set up and run this project, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
    (Note: For this sandbox environment, files are already provided.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install required Python packages:**
    ```bash
    pip install pandas numpy scikit-learn nltk
    ```

4.  **Download NLTK data:**
    Open a Python interpreter or add the following lines to your script:
    ```python
    import nltk
    nltk.download("punkt")
    nltk.download("stopwords")
    ```

## Project Structure
-   `train.csv`: Training dataset.
-   `test.csv`: Test dataset.
-   `eda.py`: Script for Exploratory Data Analysis.
-   `preprocess.py`: Script for text preprocessing and TF-IDF vectorization.
-   `model.py`: Script for building and training the Logistic Regression models.
-   `evaluate.py`: Script for evaluating the models and generating predictions.
-   `eda_results.txt`: Output file containing EDA findings.
-   `train_processed.pkl`: Pickled preprocessed training DataFrame.
-   `test_processed.pkl`: Pickled preprocessed test DataFrame.
-   `tfidf_vectorizer.pkl`: Pickled TF-IDF vectorizer.
-   `toxicity_logistic_regression_models.pkl`: Pickled trained Logistic Regression models.
-   `test_predictions.csv`: CSV file containing predictions on the test set.
-   `evaluation_results.txt`: Text file containing model evaluation metrics.

## Usage
Follow these steps to run the project:

1.  **Perform Exploratory Data Analysis (EDA):**
    ```bash
    python3 eda.py
    ```
    This will generate `eda_results.txt` with insights into the datasets.

2.  **Perform Data Preprocessing and Feature Engineering:**
    ```bash
    python3 preprocess.py
    ```
    This script will preprocess the text data, perform TF-IDF vectorization, and save the processed dataframes and the vectorizer to `.pkl` files.

3.  **Build and Train Machine Learning Models:**
    ```bash
    python3 model.py
    ```
    This will train Logistic Regression models for each toxicity label and save them to `toxicity_logistic_regression_models.pkl`.

4.  **Evaluate Models and Generate Predictions:**
    ```bash
    python3 evaluate.py
    ```
    This script will evaluate the trained models on the training data (for metrics) and generate predictions on the test set, saving them to `test_predictions.csv` and `evaluation_results.txt`.

## Results
### Evaluation Metrics
The `evaluation_results.txt` file contains the accuracy, F1 score, and ROC AUC score for each toxicity label on the training data. These metrics provide an indication of the model's performance.

### Predictions
The `test_predictions.csv` file contains the predicted toxicity labels for the comments in the `test.csv` dataset.

## VS Code Setup Guide
(Refer to `vscode_setup_guide.md` for detailed instructions.)



