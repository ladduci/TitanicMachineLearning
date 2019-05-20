import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore', DeprecationWarning)

data=pd.read_csv('train.csv')

rich_features = pd.concat([data[['Fare', 'Pclass', 'Age']],
                           pd.get_dummies(data['Sex'], prefix='Sex'),
                           pd.get_dummies(data['Embarked'], prefix='Embarked')],
                          axis=1)
						  
rich_features_no_male = rich_features.drop('Sex_male', 1)
rich_features_final = rich_features_no_male.fillna(rich_features_no_male.dropna().median())
survived_column = data['Survived']
target = survived_column.values
data_test=pd.read_csv('test.csv')

rich_features_test = pd.concat([data_test[['Fare', 'Pclass', 'Age']],
                           pd.get_dummies(data_test['Sex'], prefix='Sex'),
                           pd.get_dummies(data_test['Embarked'], prefix='Embarked')],
                          axis=1)
						  
rich_features_no_male_test = rich_features_test.drop('Sex_male', 1)
rich_features_final_test = rich_features_no_male_test.fillna(rich_features_no_male_test.dropna().median())
passngerid_column = data_test['PassengerId']


##Linear Model: Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

logreg = LogisticRegression(C=1)
scores = cross_val_score(logreg, rich_features_final, target, cv=5, scoring='accuracy')
print("Logistic Regression CV scores:")
print("min: {:.3f}, mean: {:.3f}, max: {:.3f}".format(
    scores.min(), scores.mean(), scores.max()))
	
predsLR = logreg.predict(rich_features_final_test)

resultsLR = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': predsLR})
resultsLR.to_csv(f'submissionLR.csv', index=False)

	
#Non Linear Model Random Forest Classifier	
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, rich_features_final, target, cv=5, n_jobs=4, scoring='accuracy')
print("Random Forest CV scores:")
print("min: {:.3f}, mean: {:.3f}, max: {:.3f}".format(
    scores.min(), scores.mean(), scores.max()))
rf.fit(rich_features_final, survived_column)


predsRF = rf.predict(rich_features_final_test)

resultsRF = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': predsRF})
resultsRF.to_csv(f'submissionRF.csv', index=False)

#Non Linear Model Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                subsample=.8, max_features=.5)
								
scores = cross_val_score(gb, rich_features_final, target, cv=5, n_jobs=4,
                         scoring='accuracy')
print("Gradient Boosted Trees CV scores:")
print("min: {:.3f}, mean: {:.3f}, max: {:.3f}".format(
    scores.min(), scores.mean(), scores.max()))

	rf.fit(rich_features_final, survived_column)


predsGB = gb.predict(rich_features_final_test)

resultsGB = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': predsGB})
resultsGB.to_csv(f'submission.csv', index=False)

	
