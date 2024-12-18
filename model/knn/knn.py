import pandas as pd
import numpy as np
from collections import Counter

def min_max_normalize(df):
        return (df - df.min()) / (df.max() - df.min())

def euclidean_distance(a, b):
    """Compute the Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))

def one_hot_encode(df, column):
        encoded = pd.get_dummies(df[column], prefix=column)
        return encoded

def prepaData(df, id_music = None):

    track_id_column = df['track_id']  # Conserver une copie de 'track_id'

    # Step 2: Drop unnecessary columns except 'track_name' and 'track_popularity'
    columns_to_drop = [
        'track_name', 'track_id', 'track_artist', 'track_album_id',
        'track_album_name', 'track_album_release_date', 'playlist_name',
        'playlist_id', 'popularity',
        'Popularity_Category'
    ]


    # Keep 'track_name' and 'track_popularity' for display and prediction
    track_names = df['track_name']  # Extract track names
    Popularity_Category = df['Popularity_Category']  # Extract popularity category
    df = df.drop(columns=columns_to_drop)

    # Step 3: Vectorize 'playlist_genre' and 'playlist_subgenre' (One-Hot Encoding)
    
    # Apply one-hot encoding
    playlist_genre_encoded = one_hot_encode(df, "playlist_genre")
    playlist_subgenre_encoded = one_hot_encode(df, "playlist_subgenre")

    # Convert boolean values to integers (True -> 1, False -> 0)
    playlist_genre_encoded = playlist_genre_encoded.astype(int)
    playlist_subgenre_encoded = playlist_subgenre_encoded.astype(int)

    # Combine the numerical features with the encoded categorical features
    df = df.drop(columns=["playlist_genre", "playlist_subgenre"])
    final_df = pd.concat([df, playlist_genre_encoded, playlist_subgenre_encoded], axis=1)

    # Step 4: Normalize All Features (Min-Max Normalization)
    numerical_columns = final_df.columns  # All remaining columns are numerical now
    final_df[numerical_columns] = min_max_normalize(final_df[numerical_columns])

    try:
        track_index = track_id_column[track_id_column == id_music].index[0]
        track_row = final_df.loc[track_index]
    except IndexError:
        track_row = None  # Si l'id n'est pas trouvé

    return final_df, track_names, Popularity_Category, track_row

def knn(data, query, k, categList):
    """
    Perform manual K-Nearest Neighbors algorithm.

    Args:
        data (pd.DataFrame): The dataset without the query.
        query (pd.Series): A single row to classify.
        k (int): Number of neighbors to consider.

    Returns:
        predict (str): The predicted class for the query.
    """
    distances = []
    for index, row in data.iterrows():
        distance = euclidean_distance(row.values, query.values)
        distances.append((index, distance))
    
    # Sort by distance and select the k-nearest neighbors
    distances.sort(key=lambda x: x[1])
    neighbors = [idx for idx, _ in distances[:k]]
    # get the most common class among the neighbors
    neighbors_popularity = categList.iloc[neighbors]
    counter = Counter(neighbors_popularity)
    predict = counter.most_common(1)[0][0]
    return predict
