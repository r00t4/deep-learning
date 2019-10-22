import pandas as pd
import numpy as np

trainData=pd.read_csv('./train.csv')

train=trainData.copy()
train['Age']=train['Age'].fillna(train["Age"].median())
train['Embarked']=train['Embarked'].fillna('C')
train.loc[train['Sex']=='female','Sex']=0
train.loc[train['Sex']=='male','Sex']=1
train.loc[train['Embarked']=='C','Embarked']=1
train.loc[train['Embarked']=='Q','Embarked']=2
train.loc[train['Embarked']=='S','Embarked']=3
train_x=train.loc[:,['Pclass','Age','Sex','SibSp','Parch','Embarked', 'Survived']]

print(train_x)

train_x.to_csv('train_x.csv', index=False)