import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import urllib.request

URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_PATH = "sms_spam.tsv"

def download_data():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading dataset from {URL}...")
        try:
            urllib.request.urlretrieve(URL, DATA_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download dataset. Error: {e}")
            # Ensure it doesn't crash if we have no internet and no file; we will use dummy data instead
            create_dummy_data()
    else:
        print("Dataset already exists locally.")

def create_dummy_data():
    print("Creating dummy data as fallback...")
    data = [
        ("ham", "Hey, what time are we meeting tomorrow?"),
        ("spam", "You've won a $1000 gift card! Click here now!"),
        ("ham", "Can you please review the attached document?"),
        ("spam", "URGENT! Your account has been suspended. Log in to verify."),
        ("ham", "Happy birthday! Hope you have a great day."),
        ("spam", "Get rich quick! Buy this amazing course."),
        ("spam", "Congratulations! You have been selected for a free vacation. Call now!"),
        ("ham", "Don't forget to pick up some groceries on your way home.")
    ]
    df = pd.DataFrame(data, columns=['label', 'message'])
    df.to_csv(DATA_PATH, sep='\t', index=False, header=False)


def train():
    download_data()
    
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'message'])
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    # Basic preprocessing: convert labels to binary
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    
    print(f"Dataset shape: {df.shape}")
    
    X = df['message']
    y = df['label_num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Create a pipeline combining the TF-IDF vectorizer and Naive Bayes classifier
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    print("Saving model pipeline...")
    joblib.dump(model_pipeline, 'model_pipeline.joblib')
    print("Model saved to model_pipeline.joblib")

if __name__ == "__main__":
    train()
