# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:31:03 2021

@author: BastienDeweze
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Traitement des données

df = pd.read_csv("./dataset/price_estimator.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)

# Séparation du jeu de données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Standardisation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



# Reseau de neurones

# Initialiser le reseau
classifier = Sequential()
# Premiere couche cachée
classifier.add(Dense(units=12, activation="relu", kernel_initializer="uniform", input_dim=20))

# Deuxieme couche cachée
classifier.add(Dense(units=12, activation="relu", kernel_initializer="uniform"))

# Couche de sortie
classifier.add(Dense(units=4, activation="softmax", kernel_initializer="uniform"))

# Compilation du reseau
classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entrainement du réseau
classifier.fit(X_train, y_train, batch_size=20, epochs=200)

# Prédiction
y_pred = classifier.predict(X_test)



# Test du modele

def build_classifier():
        
    classifier = Sequential()
    classifier.add(Dense(units=12, activation="relu", kernel_initializer="uniform", input_dim=20))
    classifier.add(Dense(units=12, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dense(units=4, activation="softmax", kernel_initializer="uniform"))
    classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size=20, epochs=200)
prcissions = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)



# Ajustement du modele
def build_classifier(optimizer):
    
    classifier = Sequential()
    classifier.add(Dense(units=12, activation="relu", kernel_initializer="uniform", input_dim=20))
    classifier.add(Dense(units=12, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dense(units=4, activation="softmax", kernel_initializer="uniform"))
    classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {"batch_size" : [20, 30],
              "epochs" : [200, 300],
              "optimizer" : ["adam", "rmsprop"]}


grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_param = grid_search.best_params_
best_precis = grid_search.best_score_