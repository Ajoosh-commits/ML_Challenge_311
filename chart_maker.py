import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# 1. Load the raw dataset
df = pd.read_csv('ml_challenge_dataset.csv')

print("Generating and saving graphs...")

# --- PREPARE THE DATA FOR PLOTTING ---

# Prepare Likert Columns
likert_cols = [
    'This art piece makes me feel sombre.',
    'This art piece makes me feel content.',
    'This art piece makes me feel calm.',
    'This art piece makes me feel uneasy.'
]
for col in likert_cols:
    df[col] = df[col].astype(str).apply(lambda x: float(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else np.nan)

# Prepare Price Column
def clean_price_for_plot(text):
    if pd.isna(text): return np.nan # Use NaN so 0s don't skew the average artificially
    text_no_commas = str(text).replace(',', '')
    numbers = re.findall(r'\d+\.?\d*', text_no_commas)
    return float(numbers[0]) if numbers else np.nan
df['Price'] = df['How much (in Canadian dollars) would you be willing to pay for this painting?'].apply(clean_price_for_plot)

# Prepare Intensity Column
intensity_col = 'On a scale of 1–10, how intense is the emotion conveyed by the artwork?'
df['Intensity'] = pd.to_numeric(df[intensity_col], errors='coerce')

# Prepare Text Columns (Get Top 10 Words for Food and Feelings)
stop_words = {'the', 'a', 'an', 'and', 'to', 'of', 'it', 'in', 'is', 'that', 'with', 'this', 'for', 'my', 'me', 'as', 'like', 'but', 'not', 'be', 'on', 'was', 'would', 'i', 'or', 'so'}
food_stops = stop_words.union({'painting', 'something', 'food', 'would', 'bowl', 'fresh', 'cold', 'warm'})
feel_stops = stop_words.union({'clocks', 'reminds', 'has', 'away', 'out', 'because', 'feel', 'makes', 'painting', 'feels', 'sense', 'also', 'very', 'feeling', 'about', 'there', 'gives', 'how', 'can', 'bit', 'make', 'little', 'everything', 'are', 'which'})

def get_tokens(text, stops):
    text = str(text).lower().replace('ice cream', 'ice_cream')
    return [w for w in re.findall(r'\b[a-z_]+\b', text) if w not in stops and len(w) > 2]

df['food_tokens'] = df['If this painting was a food, what would be?'].apply(lambda x: get_tokens(x, food_stops))
df['feel_tokens'] = df['Describe how this painting makes you feel.'].apply(lambda x: get_tokens(x, feel_stops))

# Find the top 10 overall words for both categories
all_foods = [word for tokens in df['food_tokens'] for word in tokens]
top_10_foods = [word for word, count in Counter(all_foods).most_common(10)]

all_feels = [word for tokens in df['feel_tokens'] for word in tokens]
top_10_feels = [word for word, count in Counter(all_feels).most_common(10)]

# Count occurrences per row for the top 10 words
for w in top_10_foods: df[f'food_{w}'] = df['food_tokens'].apply(lambda x: x.count(w))
for w in top_10_feels: df[f'feel_{w}'] = df['feel_tokens'].apply(lambda x: x.count(w))


# --- GENERATE AND SAVE PLOTS ---

# Set a consistent style
sns.set_theme(style="whitegrid")

# Graph 1: Popular Food Types per Painting
plt.figure(figsize=(12, 6))
food_cols_names = [f'food_{w}' for w in top_10_foods]
df.groupby('Painting')[food_cols_names].sum().T.plot(kind='bar', figsize=(12, 6), colormap='Set2', width=0.8)
plt.title('Occurrence of Popular Food Types by Painting', fontsize=14)
plt.ylabel('Total Occurrences')
plt.xticks(ticks=range(10), labels=top_10_foods, rotation=45)
plt.legend(title='Painting')
plt.tight_layout()
plt.savefig('graph_1_food_occurrences.png', dpi=300)
plt.close()

# Graph 2: Popular Feelings Words per Painting
plt.figure(figsize=(12, 6))
feel_cols_names = [f'feel_{w}' for w in top_10_feels]
df.groupby('Painting')[feel_cols_names].sum().T.plot(kind='bar', figsize=(12, 6), colormap='viridis', width=0.8)
plt.title('Occurrence of Popular Feeling Words by Painting', fontsize=14)
plt.ylabel('Total Occurrences')
plt.xticks(ticks=range(10), labels=top_10_feels, rotation=45)
plt.legend(title='Painting')
plt.tight_layout()
plt.savefig('graph_2_feelings_occurrences.png', dpi=300)
plt.close()

# Graph 3: Average Likert Values per Painting
plt.figure(figsize=(10, 6))
df.groupby('Painting')[likert_cols].mean().T.plot(kind='bar', figsize=(10, 6), colormap='coolwarm', width=0.7)
plt.title('Average Emotion Likert Scores by Painting', fontsize=14)
plt.ylabel('Average Score (1 to 5)')
plt.xticks(ticks=range(4), labels=['Sombre', 'Content', 'Calm', 'Uneasy'], rotation=0)
plt.legend(title='Painting', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('graph_3_likert_averages.png', dpi=300)
plt.close()

# Graph 4: Average Price per Painting
plt.figure(figsize=(8, 6))
# Dropping massive outliers (e.g., people answering billions) just for a readable chart
price_filtered = df[df['Price'] < 100000]
price_filtered.groupby('Painting')['Price'].mean().plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
plt.title('Average Amount Willing to Pay (Excluding Massive Outliers)', fontsize=14)
plt.ylabel('Price (CAD)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('graph_4_price_averages.png', dpi=300)
plt.close()

# Graph 5: Average Intensity per Painting
plt.figure(figsize=(8, 6))
df.groupby('Painting')['Intensity'].mean().plot(kind='bar', color=['#9467bd', '#8c564b', '#e377c2'], edgecolor='black')
plt.title('Average Emotional Intensity by Painting', fontsize=14)
plt.ylabel('Intensity Score (1-10)')
plt.ylim(0, 10)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('graph_5_intensity_averages.png', dpi=300)
plt.close()

# Graph 6: Categorical Modes (Rooms, Companions, Seasons)
def get_category_counts(df, col_name):
    """Helper function to un-nest comma-separated strings and count them per painting."""
    counts = {'The Persistence of Memory': Counter(), 'The Starry Night': Counter(), 'The Water Lily Pond': Counter()}
    for _, row in df.iterrows():
        painting = row['Painting']
        if pd.isna(painting): continue
        items = [i.strip() for i in str(row[col_name]).split(',') if i.strip()]
        counts[painting].update(items)
    return pd.DataFrame(counts).fillna(0)

# Generate counts
room_counts = get_category_counts(df, 'If you could purchase this painting, which room would you put that painting in?')
comp_counts = get_category_counts(df, 'If you could view this art in person, who would you want to view it with?')
seas_counts = get_category_counts(df, 'What season does this art piece remind you of?')

# Plot as a 1x3 combined figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

room_counts.plot(kind='bar', ax=axes[0], colormap='Set2')
axes[0].set_title('Preferred Room Choices', fontsize=14)
axes[0].set_ylabel('Total Occurrences')
axes[0].tick_params(axis='x', rotation=45)
axes[0].get_legend().remove()

comp_counts.plot(kind='bar', ax=axes[1], colormap='Set2')
axes[1].set_title('Preferred Companion Choices', fontsize=14)
axes[1].tick_params(axis='x', rotation=45)
axes[1].get_legend().remove()

seas_counts.plot(kind='bar', ax=axes[2], colormap='Set2')
axes[2].set_title('Associated Season Choices', fontsize=14)
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend(title='Painting', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('graph_6_categorical_modes.png', dpi=300)
plt.close()

print("Success! All 6 graphs have been saved to your directory.")
