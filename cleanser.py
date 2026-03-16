import pandas as pd
import numpy as np
import re
from collections import Counter

# --- Helper Functions ---

def extract_likert(text):
    """Extracts the leading integer from Likert scale strings."""
    match = re.search(r'\d+', str(text))
    return float(match.group()) if match else np.nan

def clean_price(text):
    """Strips currency symbols and text to return a raw float."""
    if pd.isna(text):
        return 0.0
    text_no_commas = str(text).replace(',', '')
    numbers = re.findall(r'\d+\.?\d*', text_no_commas)
    return float(numbers[0]) if numbers else 0.0

def binarize_column(df, col_name, prefix):
    """Dynamically one-hot encodes comma-separated categorical strings."""
    df[col_name] = df[col_name].fillna('')
    all_categories = set()

    for row in df[col_name]:
        items = [item.strip() for item in str(row).split(',') if item.strip()]
        all_categories.update(items)

    for category in all_categories:
        clean_cat = category.replace(' ', '_').replace('/', '_')
        df[f"{prefix}_{clean_cat}"] = df[col_name].apply(lambda x: 1 if category in str(x) else 0)

    return df.drop(columns=[col_name])

def custom_bow(df, col_name, prefix, max_features=15, custom_stop_words=None, phrase_replacements=None):
    """Custom Bag of Words implementation using raw frequency counts."""
    stop_words = {'the', 'a', 'an', 'and', 'to', 'of', 'it', 'in', 'is', 'that', 'with', 'this', 'for', 'my', 'me', 'as', 'like', 'but', 'not', 'be', 'on', 'was', 'would', 'i', 'or', 'so'}

    if custom_stop_words:
        stop_words.update(custom_stop_words)

    # Lowercase the text
    texts = df[col_name].fillna('').astype(str).str.lower()

    # Replace multi-word phrases
    if phrase_replacements:
        for phrase, replacement in phrase_replacements.items():
            texts = texts.str.replace(phrase, replacement, regex=False)

    # Tokenize
    tokenized = texts.apply(lambda x: [w for w in re.findall(r'\b[a-z_]+\b', x) if w not in stop_words and len(w) > 2])

    # Find the most common words to build the vocabulary
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(tokens) # Update with all tokens to get raw frequency

    top_words = [w for w, freq in doc_freq.most_common(max_features)]

    print(f"\n--- Top {max_features} words extracted for '{col_name}' (BoW) ---")
    print(top_words)
    print("-" * 50)

    # Count occurrences for each row
    bow_features = []
    for tokens in tokenized:
        row_dict = {}
        for word in top_words:
            row_dict[f"{prefix}_{word}"] = tokens.count(word)
        bow_features.append(row_dict)

    bow_df = pd.DataFrame(bow_features)
    return pd.concat([df.drop(columns=[col_name]), bow_df], axis=1)

# --- Core Preprocessing Pipeline ---

def preprocess_data(df, max_text_features=15):
    """Executes the full preprocessing pipeline on the raw dataframe."""
    print("Starting preprocessing pipeline...")

    # 1. Drop IDs
    if 'unique_id' in df.columns:
        df = df.drop(columns=['unique_id'])

    # 2. Target Encoding
    target_mapping = {
        'The Persistence of Memory': 0,
        'The Starry Night': 1,
        'The Water Lily Pond': 2
    }
    df['Painting_Target'] = df['Painting'].map(target_mapping)
    df = df.drop(columns=['Painting'])

    # 3. Parse Ordinal Likert Scales
    likert_cols = [
        'This art piece makes me feel sombre.',
        'This art piece makes me feel content.',
        'This art piece makes me feel calm.',
        'This art piece makes me feel uneasy.'
    ]
    for col in likert_cols:
        df[col] = df[col].apply(extract_likert)
        df[col] = df[col].fillna(df[col].median())

    # 4. Clean Price Column
    price_col = 'How much (in Canadian dollars) would you be willing to pay for this painting?'
    df[price_col] = df[price_col].apply(clean_price)

    # 5. Clean Basic Numeric Columns
    numeric_cols = [
        'On a scale of 1–10, how intense is the emotion conveyed by the artwork?',
        'How many prominent colours do you notice in this painting?',
        'How many objects caught your eye in the painting?'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # 6. Multi-Label Binarization
    df = binarize_column(df, 'If you could purchase this painting, which room would you put that painting in?', 'room')
    df = binarize_column(df, 'If you could view this art in person, who would you want to view it with?', 'with')
    df = binarize_column(df, 'What season does this art piece remind you of?', 'season')

    # 7. TF-IDF Text Vectorization
    # 1. Feelings
    feelings_stops = {'clocks', 'time', 'sky', 'reminds', 'has', 'away', 'out', 'because', 'feel', 'makes', 'painting', 'feels', 'sense', 'also', 'very', 'feeling', 'about', 'there', 'gives', 'how', 'can', 'bit', 'make', 'little', 'everything', 'are', 'which'}
    df = custom_bow(df, 'Describe how this painting makes you feel.', 'feelings',
                      max_features=max_text_features, custom_stop_words=feelings_stops)

    # 2. Food
    food_stops = {'painting', 'something', 'food', 'would', 'bowl', 'fresh', 'cold', 'warm', 'you'}
    # Glue 'ice cream' together! You can add others like 'hot dog' if you notice them.
    food_phrases = {'ice cream': 'ice_cream', 'mac and cheese': 'mac_and_cheese'}
    df = custom_bow(df, 'If this painting was a food, what would be?', 'food',
                      max_features=max_text_features, custom_stop_words=food_stops, phrase_replacements=food_phrases)

    # 3. Soundtrack
    sound_stops = {'melody', 'soundtrack', 'music', 'sound', 'very', 'sounds', 'song', 'background', 'something', 'some', 'feel', 'you', 'piece', 'maybe', 'notes', 'time', 'rhythm', 'instruments', 'have', 'there', 'tempo'}
    df = custom_bow(df, 'Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.', 'sound',
                      max_features=max_text_features, custom_stop_words=sound_stops)

    print("Preprocessing complete.")
    return df


# --- Main Execution Block ---

if __name__ == "__main__":
    input_file = 'ml_challenge_dataset.csv'
    output_file = 'processed_ml_dataset.csv'

    # Set your hyperparameter here!
    MAX_TEXT_FEATURES = 15

    # Import the data
    try:
        print(f"Loading data from {input_file}...")
        raw_df = pd.read_csv(input_file)

        # Call the processing function
        processed_df = preprocess_data(raw_df, max_text_features=MAX_TEXT_FEATURES)

        # Output logic
        processed_df.to_csv(output_file, index=False)
        print(f"Success! Processed matrix saved to {output_file}.")
        print(f"Final dataset shape: {processed_df.shape}")

    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Please ensure it is in the same directory.")
