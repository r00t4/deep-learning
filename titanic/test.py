import cv2
import glob
import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score

from model import NeuralNetwork

net = NeuralNetwork()
net.load_state_dict(torch.load('./model.pth'))

testData=pd.read_csv('./test.csv')
test=testData.copy()
test['Age']=test['Age'].fillna(test["Age"].median())
test['Embarked']=test['Embarked'].fillna('C')
test.loc[test['Sex']=='female','Sex']=0
test.loc[test['Sex']=='male','Sex']=1
test.loc[test['Embarked']=='C','Embarked']=1
test.loc[test['Embarked']=='Q','Embarked']=2
test.loc[test['Embarked']=='S','Embarked']=3
test_x=test.loc[:,['Pclass','Age','Sex','SibSp','Parch','Embarked']]


ans = list()
for i,j in test_x.iterrows():
	# print(type(np.float32(j.values)))
	# print("----")
	x = torch.from_numpy(j.values).float()
	
	pred = net(x)
	pred = pred.data.numpy()
	print(pred[0])
	if (pred[0] >= 0.5):
		ans.append(1)
	else:
		ans.append(0)
	# ans.append(pred[0].argmax())

d = { 'PassengerId': test['PassengerId'], 'Survived': ans}
df = pd.DataFrame(d)
df.to_csv('./ans.csv',index=False)
	