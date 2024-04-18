from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import preprocess_text
from data_processing import x_train,x_test, y_train, y_test
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#best_params = {'clf__C': 1, 'clf__gamma': 'scale', 'clf__kernel': 'linear', 'vec__max_features': 1500, 'vec__ngram_range': (1, 2)}

print("building and fitting the pipeline")

pipeline = Pipeline([("vec", CountVectorizer(preprocessor=preprocess_text, max_features= 1500, ngram_range= (1, 2))), ("clf", SVC(C= 1, gamma= 'scale', kernel= 'linear'))])
pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)
accuracy = pipeline.score(x_test, y_test)
print(accuracy)#0.9858 before tuning and 0.9889 after tuning

# testing the model with user input

target_variables = ["Credit reporting, repair, or other", "Debt collection", "Consumer Loan", "Mortgage"]
test_input = ["Hello I am writing this complaint to bring your attention to the fact that I have charged for wrong Debt. I have already clear all my dues so you have no right to collect debt from me. Moreover you are charging interest on debt also. This debt collection is illegal. Please stop it"]
test_prediction = pipeline.predict(test_input)
print(f"Our model has predicted \n{target_variables[test_prediction[0]]} for the given input\n {test_input}")
print(test_prediction)