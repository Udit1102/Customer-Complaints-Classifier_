from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from preprocessing import preprocess_text
from data_processing import x_train, x_test, y_train, y_test
from sklearn.model_selection import GridSearchCV

pipeline_tuning = Pipeline([("vec", CountVectorizer()), ("clf", LogisticRegression())])
param = {"clf__max_iter": [200,500], "clf__C": [0.001, 0.01, 0.1, 1, 10], "clf__penalty":["l1", "l2"], "clf__solver":["liblinear", "saga"]}
grid_search = GridSearchCV(pipeline_tuning, param, cv=5)
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
print(best_params)

'''
print("building and fitting the pipeline")

pipeline = Pipeline([("vec", CountVectorizer(preprocessor=preprocess_text)), ("clf", LogisticRegression())])
pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
'''