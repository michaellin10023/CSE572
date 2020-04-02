import numpy as np
import pandas as pd
import csv
from numpy.ma.extras import row_stack
from features import Features
from preprocess import Preprocess
from decision_tree import DecisionTree
import os


feature_matrix = ['Glucose_max','Glucose_min','Glucose_variance','velocity_max','velocity_min','velocity_variance','rms_max','rms_min','rms_variance']
target = ['target']

for i in range(1,6):

    meal_data = Preprocess(f'D:\\python projects\\CSE572\\project2\\mealData{i}.csv')
    meal_data_df = meal_data.get_dataframe()
    meal_feature = Features(meal_data_df)
    temp1_matrix = meal_feature.features_extraction()
    feature_matrix = np.row_stack((feature_matrix,temp1_matrix))
    # print(len(temp1_matrix))
    target.extend([1] * len(temp1_matrix))

    nomeal_data = Preprocess(f'D:\\python projects\\CSE572\\project2\\Nomeal{i}.csv')
    nomeal_data_df = nomeal_data.get_dataframe()
    nomeal_feature = Features(nomeal_data_df)
    temp2_matrix = nomeal_feature.features_extraction()
    feature_matrix = np.row_stack((feature_matrix,temp2_matrix))
    target.extend([0] * len(temp2_matrix))

feature_matrix = np.array(feature_matrix)
target = np.array([target])
new_feature_matrix = np.concatenate((feature_matrix, target.T), axis=1)
# print(len(new_feature_matrix[0]))

# print(len(feature_matrix))
# print(len(feature_matrix[0]))

with open('feature_matrix.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for row in feature_matrix:
        wr.writerow(row)

df = pd.DataFrame(new_feature_matrix)
df = df.iloc[1:]
# print(df.head())

decision_model = DecisionTree(df)
X_train, X_test, y_train, y_test = decision_model.split_dataset()
gini_clf = decision_model.train_gini(X_train,y_train)
entropy_clf = decision_model.train_entropy(X_train,y_train)

print("using gini index:")
gini_pred = decision_model.prediction(X_test, gini_clf)
decision_model.accuracy(y_test,gini_pred)


print("using entropy:")
entropy_pred = decision_model.prediction(X_test, entropy_clf)
decision_model.accuracy(y_test,entropy_pred)


filename = input(f"Please enter the file name:\n")
path = os.getcwd()
test_data = Preprocess(path + '\\' + filename)
test_data_df = test_data.get_dataframe()
test_feature = Features(test_data_df)
test_matrix = test_feature.features_extraction()
# print(len(test_matrix))
# print(len(test_matrix[0]))
test_pred = decision_model.prediction(test_matrix,gini_clf)
print("Results using trained model....")
print(test_pred)