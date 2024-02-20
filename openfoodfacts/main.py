import pandas as pd
from dataset_utils import (
    load_dataset,
    identify_missing_data,
    filter_columns_by_missing_percentage,
    drop_unwanted_columns,
)
from custom_label_augmentation import (
    clean_categories_column,
    separate_multi_and_single_label,
    create_topic_model,
    predict_categories,
    merge_categories_with_original_df,
)
from transformations import analyze_brands, analyze_ingredients
from preprocessing import preprocess_text
from model import vectorize_text, split_data, train_random_forest, evaluate_classifier


def main():
    # Load dataset
    df = load_dataset(path="../en.openfoodfacts.org.products.csv")

    # Data preprocessing
    df = preprocess_data(df)

    # Perform label augmentation
    df = augment_labels(df)

    # Train and evaluate the model
    train_and_evaluate_model(df)


def preprocess_data(df):
    """
    Perform initial data preprocessing steps.

    Parameters:
    - df (DataFrame): The input DataFrame.

    Returns:
    - df (DataFrame): The preprocessed DataFrame.
    """
    # Identify missing data
    missing_df = identify_missing_data(df=df)

    # Filter columns by missing percentage
    df = filter_columns_by_missing_percentage(df=df, nan_df=missing_df)

    # Drop unwanted columns
    columns_to_drop = [
        "code",
        "url",
        "creator",
        "created_t",
        "created_datetime",
        "last_modified_t",
        "last_modified_datetime",
        "last_modified_by",
        "last_updated_t",
        "last_updated_datetime",
        "last_image_t",
        "last_image_datetime",
        "main_category",
        "main_category_en",
        "image_url",
        "image_small_url",
        "image_ingredients_url",
        "image_ingredients_small_url",
        "image_nutrition_url",
        "image_nutrition_small_url",
    ]

    df = drop_unwanted_columns(df=df, columns_to_drop=columns_to_drop)

    print("Shape after dropping unwanted columns: ", df.shape)
    return df


def augment_labels(df):
    """
    Perform label augmentation.

    Parameters:
    - df (DataFrame): The input DataFrame.

    Returns:
    - df (DataFrame): The DataFrame after label augmentation.
    """
    # Clean categories column
    df = clean_categories_column(df=df)

    # Separate multi-label and single-label data
    multi_label_df, single_label_df = separate_multi_and_single_label(df=df)

    # Create topic model and predict categories
    topic_model, topics = create_topic_model(column=df.categories.unique(), nr_topics=10)
    mapping_df = create_mapping_dataframe()
    predict_categories_df = predict_categories(
        topic_model=topic_model,
        unique_categories=single_label_df.categories.unique(),
        assigned_topics_df=mapping_df,
    )
    single_label_df = merge_categories_with_original_df(
        df=single_label_df, predicted_df=predict_categories_df
    )
    print(single_label_df.target.value_counts())

    # Analyze brands and ingredients
    single_label_df = analyze_brands(df=single_label_df)
    single_label_df = analyze_ingredients(df=single_label_df)

    return single_label_df


def create_mapping_dataframe():
    """
    Create a mapping DataFrame for categories.

    Returns:
    - mapping_df (DataFrame): Mapping DataFrame for categories.
    """
    mapping_df = pd.DataFrame(
        {
            "id": [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            "target": [
                "Organic Beverages and Snacks",
                "Cheese and Bread Products",
                "Chocolate, Fruits, and Cheese",
                "International Food Items",
                "Caloric Content and Nutritional Information",
                "Integral and Supplemental Foods",
                "Dietary Supplements and Complements",
                "Energy and Ultra-Processed Products",
                "Halal and Dietary Restrictions",
                "Beverages and Instant Drinks",
            ],
        }
    )
    return mapping_df


def train_and_evaluate_model(df):
    """
    Train and evaluate the machine learning model.

    Parameters:
    - df (DataFrame): The input DataFrame.
    """
    articles, y = preprocess_text(df=df)
    print("Number of articles: ", len(articles))

    articles_tfidf, vectorizer = vectorize_text(articles)
    X_train, X_test, y_train, y_test = split_data(articles_tfidf, y, stratify=y)
    clf = train_random_forest(X_train, y_train)
    score = evaluate_classifier(clf, X_test, y_test)
    print("Accuracy score:", score)


if __name__ == "__main__":
    main()
