import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
from sklearn.preprocessing import OneHotEncoder
from iteration_utilities import duplicates


# Load the data
data = pd.read_csv('data/spotify_songs.csv')

# Drop nan values
data = data.dropna()

# drop duplicates
data = data.drop_duplicates()

# get a list of all genres
genres = data['playlist_genre'].unique()
print(genres)

# avoir un dico : {nom_musique : [genre1, genre2, genre3]}
dico = {}
for i in range(len(data)):
    if data['track_name'].iloc[i] not in dico.keys():
        dico[data['track_name'].iloc[i]] = [data['playlist_genre'].iloc[i]]
    else:
        dico[data['track_name'].iloc[i]].append(data['playlist_genre'].iloc[i])

for key in dico.keys():
    dico[key] = list(set(dico[key]))

# regarder si il y a des doublons
for key in dico.keys():
    if len(dico[key]) > 1:
        print(key, dico[key])


data = pd.read_csv('data/spotify_songs.csv')

# Drop nan values
data = data.dropna()

# drop all quality columns unless track_id
data = data.drop(columns=['track_name', 'track_artist', 'track_album_name', 'track_album_id', 'playlist_genre', 'playlist_subgenre', 'playlist_name', 'playlist_id'])
data = data.drop_duplicates()
# Trouve la longueur maximale des listes
max_length = max(len(genres) for genres in dico.values())

# Complète chaque liste avec NaN pour obtenir la même longueur
dico_padded = {key: genres + [None] * (max_length - len(genres)) for key, genres in dico.items()}

# Crée un DataFrame
pdGenre = pd.DataFrame.from_dict(dico_padded, orient='index')
print(pdGenre.head())

# vecotrize the data pdGenre
enc = OneHotEncoder()
enc.fit(pdGenre)
onehotlabels = enc.transform(pdGenre).toarray()

