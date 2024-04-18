from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import preprocess_text
from data_processing import x_train, x_test, y_train, y_test

pipeline = Pipeline([("vectorizer", CountVectorizer(preprocessor=preprocess_text)), ("clf", MultinomialNB())])
print("step 3 fitting the pipeline")

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) #97.53
print("step4 final",accuracy)

test_text = ["I have mailed the team multiple times that i have been charged for wrong debt. I have already cleared all my dues and I have also raised a request for closure of my account. Please stop charging me for wrong debt. This is illegal."]
test_pred = pipeline.predict(test_text)
print(test_pred)
