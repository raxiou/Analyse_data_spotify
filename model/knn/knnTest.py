import knn.knn as knn
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

def process_query(i, dataTest, data, k, Popularity_Category, Popularity_CategoryTest):
    """
    Fonction pour traiter une requête individuelle et retourner 1 si la prédiction est correcte, sinon 0.
    """
    query = dataTest.iloc[i]
    predict = knn.knn(data, query, k, Popularity_Category)
    return 1 if Popularity_CategoryTest.iloc[i] == predict else 0

def __main__():
    # Charger les données
    data = pd.read_csv("data/spotify_songs_with_popularity.csv", low_memory=False)

    # Préparer les données
    data, track_names, Popularity_Category = knn.prepaData(data)

    # Diviser les données en ensemble d'entraînement et de test
    dataTest = data.sample(frac=0.05)
    data = data.drop(dataTest.index)
    Popularity_CategoryTest = Popularity_Category.loc[dataTest.index]
    
    listk = [5, 7, 10]
    for k in listk:

        # Parallélisation avec joblib
        results = Parallel(n_jobs=-1)(
            delayed(process_query)(i, dataTest, data, k, Popularity_Category, Popularity_CategoryTest) 
            for i in tqdm(range(len(dataTest)), desc="Processing")
        )

        # Calcul de l'exactitude
        accuracy = sum(results) / len(dataTest)
        print(f"Accuracy for k={k}: {accuracy}")

if __name__ == "__main__":
    __main__()
