# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


titanic = pd.read_csv('titanic.csv')
info = pd.DataFrame(data=titanic)
target = info['Survived']

info = info.drop('Survived',axis=1)
info = info.drop('Name',axis=1)
info = info.drop('Ticket',axis=1)
info = info.drop('PassengerId',axis=1)
info['Cabin'] = info['Cabin'].apply(lambda x: int(0) if type(x) == float else int(1))
info['Embarked'] = info['Embarked'].fillna('S')
info['Embarked'] = info['Embarked'].map({'Q':0,'S':1,'C':2}).astype(int)
info['Sex'] = info['Sex'].map({'female':0,'male':1}).astype(int)
info['Age'] = info['Age'].replace(np.nan,info['Age'].mean())
info['Age'] = pd.cut(info['Age'],[0,8,16,23,36,42,56,80],labels = ['1','2','3','4','5','6','7'])


x_train,x_test,y_train,y_test = train_test_split(info,target,test_size=0.1)
clf = DecisionTreeClassifier(max_depth = 3,random_state = 0)

clf.fit(x_train,y_train)

prediccion = clf.predict(x_test[0:10])

for x in prediccion:
     print('Vivo' if x == 1 else'Muerto')

max_depth_range = list(range(1,6))
accuracy = []
for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth = depth,random_state=0)
    clf.fit(x_train,y_train)
    score = clf.score(x_test,y_test)
    accuracy.append(score)
print(accuracy)