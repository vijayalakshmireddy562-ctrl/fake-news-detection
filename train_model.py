
import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_dataset(file_name, label):
    if not os.path.exists(file_name):
        raise FileNotFoundError(
            f"Could not find {file_name}. "
            "Place it in the same folder as train_model.py."
        )

    df = pd.read_csv(file_name)

    # Some versions of these datasets contain a title and text column.
    if "text" not in df.columns:
        raise ValueError(
            f"{file_name} must contain a 'text' column. "
            f"Found columns: {list(df.columns)}"
        )

    if "title" in df.columns:
        df["title"] = df["title"].fillna("").astype(str)
        df["text"] = df["title"] + " " + df["text"].fillna("").astype(str)
    else:
        df["text"] = df["text"].fillna("").astype(str)

    df = df[["text"]].copy()
    df["label"] = label
    return df


def main():
    print("Loading datasets...")

    fake_news = load_dataset("Fake.csv", 0)
    real_news = load_dataset("True.csv", 1)

    data = pd.concat([fake_news, real_news], ignore_index=True)
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"].str.len() > 0]
    data = data.drop_duplicates(subset=["text"])

    if data["label"].nunique() < 2:
        raise ValueError("Both fake and real news datasets are required.")

    print(f"Total articles: {len(data)}")
    print("Preparing training and testing data...")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )

    print("Training model. Please wait...")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["FAKE", "REAL"]
    ))

    joblib.dump(model, "fake_news_model.pkl")
    print("\nModel saved as fake_news_model.pkl")
    print("Training complete!")


if __name__ == "__main__":
    main()
