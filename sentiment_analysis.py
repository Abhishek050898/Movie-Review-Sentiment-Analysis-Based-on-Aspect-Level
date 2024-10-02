
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load IMDb Dataset
def load_data(file_path):
    """
    Load the dataset from a CSV file.
    The dataset should have two columns: 'review' and 'sentiment'
    """
    data = pd.read_csv(file_path)
    return data['review'], data['sentiment']

# Preprocessing
def preprocess_data(reviews):
    """
    Perform text preprocessing, such as removing punctuation, 
    handling case, and converting the text to a numerical representation.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(reviews)
    return X

# Train Model
def train_model(X_train, y_train, model_type="logistic"):
    """
    Train a sentiment analysis model based on the given model type.
    Available models: 'logistic', 'naive_bayes', 'svm'
    """
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "naive_bayes":
        model = MultinomialNB()
    elif model_type == "svm":
        model = SVC(kernel='linear', probability=True)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set and print the classification report and confusion matrix.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Hyperparameter Tuning (Optional)
def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV to optimize the Logistic Regression model.
    """
    param_grid = {'C': np.logspace(-4, 4, 20), 'penalty': ['l2']}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best Hyperparameters:", grid.best_params_)
    return grid.best_estimator_

# Main Execution
if __name__ == "__main__":
    # Load and preprocess the data
    reviews, sentiments = load_data("path/to/your/imdb_dataset.csv")  # Update the path
    X = preprocess_data(reviews)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)

    # Train the model (logistic regression as default)
    model = train_model(X_train, y_train, model_type="logistic")

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Optionally, perform hyperparameter tuning
    # tuned_model = hyperparameter_tuning(X_train, y_train)
    # evaluate_model(tuned_model, X_test, y_test)
