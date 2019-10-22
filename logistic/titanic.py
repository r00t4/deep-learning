import pandas as pd
import numpy as np

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
train_x=train.loc[:,['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']]
train_y=train['Survived']

value = ['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked', '']

theta = np.zeros(8)
lr = 0.001

v = np.zeros(8)

columns = len(train_x)
# value = { '': 0, 'Pclass': 1,'Age': 2,'Sex': 3,'SibSp': 4,'Parch': 5,'Fare': 6,'Embarked': 7}

for _ in range(1000):
    dt = np.zeros(8)

    for i in range(columns):
        z = 0
        for j, val in enumerate(value):
            if val == '' :
                z += theta[j]
            else:
                z += train_x[val][i]*theta[j]
        
        a = 1/(1+np.exp(-z))

        diff = a - train_y[i]

        for j,val in enumerate(value) :
            if val == '':
                dt[j] += diff
            else:
                dt[j] += diff * train_x[val][i]

    for i in range(dt.size):
        dt[i] /= len(train_x)

    for i in range(v.size):
        v[i] = 0.99*v[i] + (1-0.99)*dt[i]

    for i in range(theta.size):
        theta[i] -= lr*v[i]


print("test starting")

testData=pd.read_csv('./test.csv')
test=testData.copy()
test['Age']=test['Age'].fillna(test["Age"].median())
test['Fare']=test['Fare'].fillna(test["Fare"].median())
test.loc[test['Sex']=='female','Sex']=0
test.loc[test['Sex']=='male','Sex']=1
test.loc[test['Embarked']=='C','Embarked']=1
test.loc[test['Embarked']=='Q','Embarked']=2
test.loc[test['Embarked']=='S','Embarked']=3
test_x=test.loc[:,['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']]
test_y=gender_submission['Survived']


ans = list()
for i in range(len(test_x)):
    cnt = 0
    for j, val in enumerate(value):
        if val == '':
            cnt += theta[j]
        else:
            cnt += test_x[val][i] * theta[j]
    if 1/(1+np.exp(-cnt)) >= 0.5:
        ans.append(1)
    else :
        ans.append(0)

d = { 'PassengerId': test['PassengerId'], 'Survived': ans}
df = pd.DataFrame(d)
df.to_csv('./ans.csv',index=False)
print(theta)