from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def vectorize_text(articles):
    """
    Vectorize text data using TF-IDF.

    Parameters:
    - articles (list): List of preprocessed text articles.

    Returns:
    - articles_tfidf (sparse matrix): TF-IDF vectors of the articles.
    - vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, stop_words="english")
    articles_tfidf = vectorizer.fit_transform(articles)
    return articles_tfidf, vectorizer


def split_data(X, y, test_size=0.20, random_state=42, stratify=None):
    """
    Split the data into train and test sets.

    Parameters:
    - X (array-like): Features.
    - y (array-like): Target.
    - test_size (float): Size of the test set. Default is 0.20.
    - random_state (int): Random state for reproducibility. Default is 42.
    - stratify (array-like): Specify if stratified sampling is desired. Default is None.

    Returns:
    - X_train, X_test, y_train, y_test: Train and test data splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators=100, random_state=0):
    """
    Train a Random Forest classifier.

    Parameters:
    - X_train (array-like): Training features.
    - y_train (array-like): Training target.
    - n_estimators (int): Number of trees in the forest. Default is 100.
    - random_state (int): Random state for reproducibility. Default is 0.

    Returns:
    - clf: Trained Random Forest classifier.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test, y_test):
    """
    Evaluate the classifier on the test set.

    Parameters:
    - clf: Trained classifier.
    - X_test (array-like): Test features.
    - y_test (array-like): Test target.

    Returns:
    - score: Accuracy score of the classifier on the test set.
    """
    score = clf.score(X_test, y_test)
    return score
