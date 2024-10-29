# Maitrisez l'apprentissage supervisé


### Contexte

Ce repo centralise toutes les ressources (code de screencasts, énoncés et corrigés d'exercices) du projet filé lié au cours "Maitrisez l'apprentissage supervisé". La sturcture des différents dossiers est comme suit

.
├── exerices                # Notebooks énoncé et corrigé pour chaque chapitre du cours où un exerice est prévu
├── screencasts             # Code présenté pendant chaque screencast
└── screenshots             # Code permettant de générer certains screenshots du cours



### Comment utiliser la donnée

Vous pouvez télécharger tous les jeux de données dont vous aurez besoin via ce zip. Vous y trouverez :
* **transactions_immobilieres.parquet** : le fichier de données principale qui servira pour toute la partie 1 
* **transactions_par_ville.parquet** : fichier de données contenant des données agrégées à la maille ville, très utilisé dans toutes les parties du cours pour le calcul de features ou de la target
* **transactions_post_feature_engineering.parquet** : fichier de donnéesutilisé dans la partie 2 et 3, où le feature engineering est considéré acquis. 
* **transactions_extra_infos.parquet** : fichier de données contenant des informations sur les observations qui ne peuvent pas être utilisées comme features (id_transaction, date_transaction) mais qui sont utiles pour des analyses exploratoires. A utiliser avec le fichier transactions_post_feature_engineering.parquet
* **features_used.json** : fichier sous forme de liste contenant toutes les features utilsiées dans la modélisation. 
* **categorical_features_used.json** : sous-ensemble du ficher précédent, avec uniquement les features qualitatives 

Les différentes données sont soit au format parquet (format plus pratique que le csv et utilisable avec Pandas ou Polars) soit au format JSON quand il s'agit de listes de features.


Dans le cas où vous souhaiteriez reconstituer la donnée transactions_immobilieres.parquet, vous pouvez partir des données brutes dans ce zip et utiliser le script preprocessing.py.

Pour ne pas alourdir le repo, les données et les modèles sauvegardés via Mlflow n'ont pas été chargés ici. Concernant ces derniers, vous avez le code pour les reconstruire. 

### Comment utiliser le code

Ce repo utilise Poetry pour le package management et la création d'un environnement virtuel. le fichier pyproject.toml contient toutes les versions de packages utilisées dans le cours. 

Pour des raisons de conflits de dépendences avec Mlflow, le package BentoML (utilisé pour la partie 3 chaptire 2) ne figure pas parmi les dépendences installées. Toutefois, vous trouverez en commentaire dans le fichier pyprojet.toml la version exacte qui a été utilisé pour le code des screencasts et des exercices.

