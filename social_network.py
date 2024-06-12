#Importing neccesary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Reading data
social_data=pd.read_csv('Social_Network_Ads.csv')
social_data
social_data=social_data.drop('User ID',axis=1)
#Preprocessing Data
le=LabelEncoder()
social_data['Gender']=le.fit_transform(social_data['Gender'])
#Splitting train set and test set
X=social_data[['Age','EstimatedSalary','Gender']]
y=social_data['Purchased']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=.25)
#Train the model using RandomClassifier
rf_clf=RandomForestClassifier(n_estimators=100,random_state=42)
rf_clf.fit(X_train,y_train)
#creating a pickle file
import pickle
with open('model.pkl','wb') as model_file:
    pickle.dump(rf_clf,model_file)