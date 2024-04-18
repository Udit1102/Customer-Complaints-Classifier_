from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import preprocess_text
from data_processing import x_train,x_test, y_train, y_test, x, y
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA

'''
#Trying to tune hyper parameters for better performance
param = {
    'vec__max_features': [1000,1500, 2000],
    'vec__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1, 10],               
    'clf__kernel': ['linear', 'rbf'],     
    'clf__gamma': ['scale', 'auto']       
}

pipeline = Pipeline([("vec", CountVectorizer(preprocessor=preprocess_text)), ("clf", SVC())])
grid_search = GridSearchCV(pipeline, param, cv=5, n_jobs=-1, return_train_score=True, scoring="accuracy")
grid_search.fit(x_train, y_train)
best_param = grid_search.best_params_
print(best_param)
'''

'''#trying to reduce dimensionality using PCA
vec = CountVectorizer(preprocessor=preprocess_text, max_features=1500)
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

pca = PCA(n_components=4)
x_pca = pca.fit_transform(x_vec.toarray())
print(sum(pca.explained_variance_ratio_))

clf = SVC(C= 1, gamma= 'scale', kernel= 'linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
'''

