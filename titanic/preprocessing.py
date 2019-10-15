import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score

trainData=pd.read_csv('./train.csv')
gender_submission=pd.read_csv('./gender_submission.csv')

train=trainData.copy()
train['Age']=train['Age'].fillna(train["Age"].median())
train['Embarked']=train['Embarked'].fillna('C')
train.loc[train['Sex']=='female','Sex']=0
train.loc[train['Sex']=='male','Sex']=1
train.loc[train['Embarked']=='C','Embarked']=1
train.loc[train['Embarked']=='Q','Embarked']=2
train.loc[train['Embarked']=='S','Embarked']=3
train_x=train.loc[:,['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked', 'Survived']]

train_x.to_csv('train_x.csv', index=False)