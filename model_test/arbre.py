import pandas as pd
from CHAID import Tree
import warnings

def predict_chaid_rules(data, rules):
    # rules est une liste de tuples (condition, prédiction)
    predictions = []
    for _, row in data.iterrows():
        for condition, prediction in rules:
            if condition(row):
                predictions.append(prediction)
                break
        else:
            predictions.append(None)  # Si aucune règle ne s'applique
    return predictions


# Suppress CHAID graph warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the data
data = pd.read_csv('data/spotify_songs_with_popularity.csv', low_memory=False)

#enlever la ligne 711MglQhnhOF3UAiAs9A59
musiqueATest = data[data['track_id'] != '711MglQhnhOF3UAiAs9A59']

# Drop unnecessary columns
columns_to_drop = [
    'track_id', 'track_name', 'track_artist', 'track_album_name',
    'track_album_id', 'playlist_genre', 'playlist_subgenre',
    'playlist_name', 'playlist_id'
]
data = data.drop(columns=columns_to_drop)

musiqueATest = musiqueATest.drop(columns=columns_to_drop)

# Verify that 'Popularity_Category' exists in the data
if 'Popularity_Category' not in data.columns:
    raise ValueError("Column 'Popularity_Category' is missing in the dataset.")

# Map columns to data types for CHAID
data_types = dict(zip(data.columns, ['nominal'] * len(data.columns)))

# enlever les ligne 711MglQhnhOF3UAiAs9A59
# Create the CHAID tree
try:
    print("Generating CHAID tree...")
    tree = Tree.from_pandas_df(data, data_types, 'Popularity_Category')
    print("CHAID tree generated successfully.")
    # test de l'arbre
    print("Testing the tree...")
    rules = tree.print_tree()
    predictions = predict_chaid_rules(musiqueATest, rules)
    print("Tree test successful.")
    print("Predictions:")

except Exception as e:
    print(f"Error generating CHAID tree: {e}")
