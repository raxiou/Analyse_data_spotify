# Analyse de données musical et creation d'un model de prédiction
## Se projet se compose de 6 dossiers
* /data le dossier où les csv sont stocké
* /img le dossier où les images généré sont stocké
* /model_test le dossier où l'on a fait des essai pour les différent model
* /test le dossier où les essai pour l'analyse ont etait fait
* /analyse_data le dossier où les analyse et les image pour l'analyse sont fait
* /model le dossier central de notre dossier, c'est le dossier où le model est fait et où l'on peut lancer le modèle

## bibliotheque à installer
pour lancer le modèle il faut installer différente librairie python qui sont :
pandas
sys
numpy
collections
tqdm
joblib

## lancer le modèle
pour lancer le le modèle dans le dossier model il y a un fichier main qui permet de lancer l'app
dans le dossier knn il y a 2 fichier, knn.py qui est le code du modèle et knnTest qui permet de lancer les test du model

## Protocole de Test 
pour tester notre modèle nous séparons notre dataset en en 2 partis (95/5) puis nous testons notre modèle sur les 5% de données de teste avec comme entrés les 95% de données restante, nous faisons ca pour différent paramètre puis nous calculons l'accurancy.
vous pouvez lancer les tests en lancant le programme main.
