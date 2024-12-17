import pandas as pd
import plotly.express as px

musicdf = pd.read_csv('./data/spotify_songs.csv')

#group by genres
sumOfGenres = musicdf.groupby(['playlist_genre']).count()
sumOfSubgenres = musicdf.groupby(['playlist_subgenre']).count()
print(sumOfGenres["track_id"])
print(sumOfGenres.index)

fig1 = px.pie(sumOfGenres, values='track_id', names=sumOfGenres.index, title='Cheese Diagram of Genres in a Spotify Playlist')
fig2 = px.pie(sumOfSubgenres, values='track_id', names=sumOfSubgenres.index, title='Cheese Diagram of Sub-Genres in a Spotify Playlist')
fig1.show()
fig2.show()