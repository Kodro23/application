"""
Prediction de la survie d'un individu sur le Titanic
"""

import os
from dotenv import load_dotenv
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from src.models.models import create_pipeline, evaluate_model, split_train_test

MAX_DEPTH = None
MAX_FEATURES = "sqrt"


# ENVIRONMENT CONFIGURATION ---------------------------

parser = argparse.ArgumentParser(description="Paramètres du random forest")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres"
)
args = parser.parse_args()



# API TOKEN
load_dotenv()
JETON_API = os.environ.get("JETON_API", "")

if JETON_API.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")


# IMPORT ET EXPLORATION DONNEES --------------------------------

TrainingData = pd.read_csv("./data/data.csv")


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
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.+
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")
X_train, X_test, y_train, y_test = split_train_test(TrainingData, test_size=0.2, train_path="train.csv", test_path="test.csv")


# PIPELINE ----------------------------

# Définition des variables
pipe = create_pipeline(n_trees=args.n_trees)
pipe.fit(X_train, y_train)
# ESTIMATION ET EVALUATION ----------------------


# score
rdmf_score = pipe.score(X_test, y_test)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")

print(20 * "-")
print("matrice de confusion")
print(evaluate_model(pipe, X_test, y_test))
