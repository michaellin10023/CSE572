import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics


class DecisionTree:
    def __init__(self,df):
        self.df = df

    def split_dataset(self):

        X = self.df.values[:,:-1]
        y = self.df.values[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=200)

        return X_train, X_test, y_train, y_test

    def train_gini(self, X_train, y_train):

        # X_train, X_test, y_train, y_test = self.split_dataset()
        clf = tree.DecisionTreeClassifier(criterion="gini",max_depth=4,min_samples_leaf=5,random_state=100)
        clf.fit(X_train, y_train)
        return clf

    def train_entropy(self, X_train, y_train):

        # X_train, X_test, y_train, y_test = self.split_dataset()
        clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=4,min_samples_leaf=5,random_state=100)
        clf.fit(X_train, y_train)
        return clf

    def prediction(self, X_test, clf):

        y_pred = clf.predict(X_test)
        # print("Predicted values:")
        # print(y_pred)
        return y_pred

    def accuracy(self, y_test, y_pred):

        print("Confusion Matrix: ",
        metrics.confusion_matrix(y_test, y_pred))

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        print("Report:",metrics.classification_report(y_test,y_pred))








