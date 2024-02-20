import pandas as pd
from tqdm import tqdm


def analyze_brands(df):
    """
    Analyze brands in the DataFrame and classify them based on count.

    Parameters:
    - df (DataFrame): The input DataFrame.

    Returns:
    - df (DataFrame): The DataFrame with brand analysis results.
    """
    brands_df = df.brands.value_counts().reset_index()

    # Merge brand counts into the main DataFrame
    # brand_df = brands_count_df.rename(columns={"brands": "count", "index": "brands"})
    df = pd.merge(left=df, right=brands_df, on=["brands"], how="left")

    # Classify brands based on count
    df.loc[df["brands"].isna(), "brand_type"] = "not classified"
    df.loc[df["count"] == 1, "brand_type"] = "slow moving"
    df.loc[(df["count"] > 1) & (df["count"] <= 3), "brand_type"] = "average moving"
    df.loc[(df["count"] > 3), "brand_type"] = "fast moving"
    df.drop("count", axis=1, inplace=True)

    print("Brand Type: ", df.brand_type.value_counts())
    return df


def analyze_ingredients(df):
    """
    Analyze ingredients in the DataFrame and standardize their representation.

    Parameters:
    - df (DataFrame): The input DataFrame.

    Returns:
    - df (DataFrame): The DataFrame with standardized ingredients representation.
    """
    ingredients_columns = [
        "fat_100g",
        "saturated-fat_100g",
        "trans-fat_100g",
        "cholesterol_100g",
        "carbohydrates_100g",
        "sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "salt_100g",
        "sodium_100g",
        "vitamin-a_100g",
        "vitamin-c_100g",
        "calcium_100g",
        "iron_100g",
    ]

    for icolumn in tqdm(ingredients_columns):
        df.loc[df[icolumn].notna(), icolumn] = icolumn.split("_")[0]

    return df
