<<<<<<< HEAD
"""
Prediction de la survie d'un individu sur le Titanic
"""

import os
from dotenv import load_dotenv
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

=======

import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
>>>>>>> d339c3d2cfa492cc9bb2f0a9402fab78aca7a387
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

<<<<<<< HEAD


=======
JETON_API = "$trotskitueleski1917"


#Load dataset
TrainingData = pd.read_csv("./data.csv")
TrainingData.head()

#Clean dataset

TrainingData["Ticket"].str.split("/").str.len()
TrainingData["Name"].str.split(",").str.len()


## Un peu d'exploration et de feature engineering
TrainingData.isnull().sum()
### Statut socioéconomique

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6)
)  # layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")


### Age

sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()

## Encoder les données imputées ou transformées.


numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

# Paramétrer l'arbre de décision
parser = argparse.ArgumentParser(description="nombres d'arbres pour le random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Un nombre d'arbres pour le random forest"
)
parser.add_argument(
    "--max_depth", type=int, default=None, help="Profondeur maximale des arbres"
)
parser.add_argument(
    "--max_features", type=str, default="sqrt", help="Le nombre de features à considérer pour le meilleur split"
)
args = parser.parse_args()
print("nombre d'arbres", args.n_trees,
      "\nmax depth", args.max_depth,
      "\nmax_features", args.max_features
      )
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=20)),
    ]
)


# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")
>>>>>>> d339c3d2cfa492cc9bb2f0a9402fab78aca7a387

# ENVIRONMENT CONFIGURATION ---------------------------

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres",
)
parser.add_argument(
    "--max_depth", type=int, default=None, help="max depth",
)
parser.add_argument(
    "--max_features", type=str, default="sqrt", help="max features",
)
args = parser.parse_args()


print("n_trees:", args.n_trees, "\nmax_depth:", args.max_depth, "\nmax_features:", args.max_features)

# API TOKEN
load_dotenv()
JETON_API = os.environ["JETON_API"]
if JETON_API.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")


# IMPORT ET EXPLORATION DONNEES --------------------------------

TrainingData = pd.read_csv("data.csv")


TrainingData["Ticket"].str.split("/").str.len()
TrainingData["Name"].str.split(",").str.len()

TrainingData.isnull().sum()

# Statut socioéconomique
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")

# Age
sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()


# SPLIT TRAIN/TEST --------------------------------

# On _split_ notre _dataset_ d'apprentisage
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.

y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_train, y_train]).to_csv("train.csv")
pd.concat([X_test, y_test]).to_csv("test.csv")

<<<<<<< HEAD

# PIPELINE ----------------------------

# Définition des variables
numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

# Variables numériques
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

# Variables catégorielles
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

# Pipeline
n_trees = args.n_trees
MAX_DEPTH = args.max_depth
MAX_FEATURES = args.max_features
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=MAX_DEPTH,
            max_features=MAX_FEATURES
        )),
    ]
)


# ESTIMATION ET EVALUATION ----------------------

pipe.fit(X_train, y_train)

# score
=======
# Random Forest
# Ici demandons d'avoir 20 arbres
pipe.fit(X_train, y_train)

# calculons le score sur le dataset d'apprentissage et sur le dataset de test (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
>>>>>>> d339c3d2cfa492cc9bb2f0a9402fab78aca7a387
rdmf_score = pipe.score(X_test, y_test)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")
<<<<<<< HEAD

=======
>>>>>>> d339c3d2cfa492cc9bb2f0a9402fab78aca7a387
print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_test)))
