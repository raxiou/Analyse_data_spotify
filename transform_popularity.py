import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('data/spotify_songs.csv')

# Drop nan values
data = data.dropna()

# drop duplicates
data = data.drop_duplicates()

def transformation(data):
    data.loc[data['track_popularity'] > 95, 'popularity'] = 'Mondial Hit'
    data.loc[(data['track_popularity'] > 75) & (data['track_popularity'] < 95), 'Popularity_Category'] = 'Very Popular'
    data.loc[(data['track_popularity'] > 50) & (data['track_popularity'] < 75), 'Popularity_Category'] = 'Popular'
    data.loc[(data['track_popularity'] > 25) & (data['track_popularity'] < 50), 'Popularity_Category'] = 'Not Very Popular'
    data.loc[(data['track_popularity'] > 10) & (data['track_popularity'] < 25), 'Popularity_Category'] = 'Niche'
    data.loc[data['track_popularity'] < 10, 'Popularity_Category'] = 'Unknown'

    return data

data = transformation(data)
print(data['Popularity_Category'].value_counts())

# remove columns track_popularity
data = data.drop(columns=['track_popularity'])

# Save the data
data.to_csv('data/spotify_songs_with_popularity.csv', index=False)