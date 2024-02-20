from nltk.corpus import stopwords
from tqdm import tqdm

# import nltk
# nltk.download("stopwords")


def preprocess_text(df):
    """
    Preprocess text data in a DataFrame column by removing stopwords and non-alphabetic characters.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column_name (str): The name of the column containing text data.

    Returns:
    - processed_articles (list): List of preprocessed text articles.
    """

    df = df.astype(str)
    df.fillna("", inplace=True)
    y = df.pop("target")
    df.drop(["categories_tags", "categories_en", "categories"], axis=1, inplace=True)
    df["data"] = df.apply(" ".join, axis=1)

    stop = stopwords.words("english")
    articles = df["data"].tolist()

    processed_articles = []
    for article in tqdm(articles):
        # Remove stop words and non-alphabetic characters, and convert to lowercase
        article = " ".join(
            [word for word in article.lower().split() if word not in stop and word.isalpha()]
        )
        processed_articles.append(article)

    return processed_articles, y
