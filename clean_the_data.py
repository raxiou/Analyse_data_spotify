import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as py

# Load the data
data = pd.read_csv('data/spotify_songs.csv')

# Drop nan values
data = data.dropna()

# drop duplicates
data = data.drop_duplicates(subset=['track_id', 'track_album_id', 'playlist_id'], inplace=True)

