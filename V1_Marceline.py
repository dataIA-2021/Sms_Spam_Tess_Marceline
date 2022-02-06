# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:05:24 2022

@author: marceline 
"""

#librairie traitement du csv(dataframe)
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
import collections
from collections import OrderedDict
cmaps = OrderedDict()
import re
import string
from string import digits


#librairies graphiques
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from collections import OrderedDict
cmaps = OrderedDict()
from wordcloud import WordCloud, STOPWORDS

#librairies machine learning
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline


def nettoyage_data ():
#lecture des données sms_spam
    spam = pd.read_csv("https://raw.githubusercontent.com/dataIA-2021/Sms_Spam_Tess_Marceline/main/spam.csv",encoding ='ISO-8859-1')
#Supression des 3 dernières colonnes
    spam1=spam.drop(columns=["Unnamed: 2", "Unnamed: 3","Unnamed: 4"])
#renommer les colonnes
    spam1.columns = ['infos','messages']
#nbre de messages ham / spam 
    #resultat = spam.v1.value_counts()
    return spam1
#Résultat des spam/ham
plt.figure(figsize=(10, 5))
sns.countplot(x="infos", data=nettoyage_data())
plt.xticks(rotation=90)
plt.show()

#Mise en place des features
spam1=nettoyage_data()

def data_feat(spam1):
#AJOUT DE FEATURES
#colonne longueur
    spam1["longueur"] = spam1["messages"].apply(len)
    spam1=spam1.assign(num_tel=0)

    for i in spam1.index:
        list_telephone_5 = re.findall(r"\d{5}", spam1["messages"][i])
        if len(list_telephone_5) >= 1:
            spam1['num_tel'][i] = 1        
#Colonne spam (1,0)
    spam1["spam"] = spam1['infos'].apply(lambda x:1 if x=='spam' else 0)
    return spam1

spam1= data_feat(spam1)

#graphique
plt.figure(figsize=(14,4))
sns.distplot(data_feat(spam1).longueur.values,80)
plt.show()

#Plot pour les longueurs de messages Spam/Ham
sns.set_style("darkgrid")
sns.set(rc = {'figure.figsize' : (18,6)})
data_feat(spam1).hist(column = 'longueur', by = 'infos', bins = 70, edgecolor = 'k')
plt.show()

#data_feat(spam1).groupby('infos').describe()
      
  

#Récupération des messages spam
def data_spam(spam1):
#selection de la colonne ham et transformation en minuscule
    spam2 = spam1['messages'].loc[spam1.infos=='spam']
    spam2 = ' '.join(spam2.str.lower())
    return spam2
spam2 = data_spam(spam1)

def data_tri(spam2):
#Vérification des mots avec un barchart.
     
    stoplist =stopwords.words('english')
    #stoplist =stopwords.add('u','2', 'ur','4')
    filtered_words = [word for word in spam2.split() if word not in stoplist]
    counted_words = collections.Counter(filtered_words)    
#graphique avec le stopwords des 20 mots les plus fréquents
    words = []
    counts = []
    for letter, count in counted_words.most_common(20):
        words.append(letter)
        counts.append(count)
    return words, counts


#Barchart des mots qui ont les plus grandes occurences
colors = cm.summer(np.linspace(0, 1, 10))
rcParams['figure.figsize'] = 10, 10
words, counts = data_tri(spam2)
plt.title('mots les plus fréquents dans les messages spam')
plt.xlabel('nombres d occurences')
plt.ylabel('Mots')
plt.barh(words, counts, color=colors)
plt.show()

#graphique nuage de mots
stoplist =stopwords.words('english')
wordcloud = WordCloud(stopwords=stoplist, background_color="white", max_words=1000).generate(spam2)
rcParams['figure.figsize'] = 10, 10
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#Mise en place du prépocessing
spam4 = data_feat(spam1) 
def prepocessing(spam1):
#Extraction de texte
    spam1=spam1.assign(mot_spam=0)
    for i in spam1.index:
        list_motspam = re.compile("free|call|price|win|won|new|now|cash|text|txt|nokia|urgent|150p")
        list_motspam_find = list_motspam.findall(spam1["messages"][i])
        if len(list_motspam_find) >= 1:
            spam1['mot_spam'][i] = 1
    X = spam1.drop(['infos', 'messages','spam'], axis=1)
    Y = spam1['infos']
    return X, Y

X,Y = prepocessing(spam4)
Y = label_binarize(Y, classes=['ham', 'spam'])
# Split et fit

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

#---------------------Modèle RandomForest
#faire du tuning (recherche des hyperparamètres du classifier RandomForest)    
param_grid = {
                            'model__criterion': ["gini", "entropy"],
                            'model__n_estimators': [90, 100, 115, 130],
}
transformer_num = ColumnTransformer(transformers=[
                                                      ('Scaling', RobustScaler(), ['longueur'])
                                                      ]
                                         )
transformer_num = None
pipe = Pipeline(steps=[
                             ('transformer', transformer_num),
                             ('model', RandomForestClassifier())
                             ]
                      )
# Declare the Grid Search method
grid = GridSearchCV(pipe, param_grid, scoring = 'accuracy', cv=StratifiedKFold(n_splits = 5, shuffle = True, random_state = 123))
grid.fit(X_train, Y_train)
rf_train=grid.score(X_train, Y_train)
rf_test=grid.score(X_test, Y_test)
print(rf_train * 100)
print(rf_test * 100)

#création et le résultat de la matrice de confusion
plot_confusion_matrix(grid, X_test, Y_test,
                                 cmap=plt.cm.summer, # other color palettes : https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
                                 normalize=None)

#---------------------Modèle KNeighborsClassifier

#faire du tuning (recherche des hyperparamètres du classifier KNeighborsClassifier)    
param_grid = {
                            'model__n_neighbors': [3, 5, 7],
                            'model__weights': ['uniform', 'distance'],
                            
                            }
                           
pipe = Pipeline(steps=[
                             ('model',KNeighborsClassifier())])
                             
# déclare la méthode de Gridsearch
grid = GridSearchCV(pipe, param_grid, scoring = 'accuracy', cv=StratifiedKFold(n_splits = 5, shuffle = True, random_state = 123))
grid.fit(X_train, Y_train)

knn_train=grid.score(X_train, Y_train)
knn_test=grid.score(X_test, Y_test)
print(knn_train * 100)
print(knn_test * 100)


plot_confusion_matrix(grid, X_test, Y_test,
                                 cmap=plt.cm.summer, # other color palettes : https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
                                 normalize=None)

