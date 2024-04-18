#Customer-Complaints-Classifier
**Requirements
scikit-learn, NLTK, pandas

**Dataset
https://catalog.data.gov/dataset/consumer-complaint-databasehttps://catalog.data.gov/dataset/consumer-complaint-database

**Steps 
1. Load the dataset using pandas and analyze the data.
2. Cleaned the data and reduced the size of dataset without losing essence because of hardware limitations.
3. Preprocessed the data using NLTK and pandas.
4. Converted the data into numerical form using CountVectorizer.
5. Tried to reduce the dimensionality of the data using PCA but resulted in poor accuracy.
6. Initially selected 3 models for training- MultinomialNB (97% accuracy), SVM (98+% accuracy), LogisticRegression (98% accuracy).
7. After optimizing all the models SVM outperformed others with highest accuracy of about 98.89% and post tuning the hyper parameters accuracy and computation time also reduced significantly focusing only on top 1500 features instead of whole dataset features.
8. Final predictions were made with almost 99% accuracy and a more generalized model.

**Files
1. complaints.csv dataset- original, refer to dataset link above
2. cleaned_complaints_dataset.csv- reduced and cleaned dataset
3. complaints_data_selector.py- code to clean and reduce the original dataset
4. target_variable_function_converter.py- function to convert the 4 classes of complaints into numerical form
5. data_processing.py- splitting the data into input and target variables
6. preprocessing.py- preprocess the text and deploy NLP techniques
7. multinomialnb_raw_model.py- MultinomialNB model
8. logistic_regression_raw_model.py- Logistic Regression model
9. svm_final_model.py and svm_optimizing.py- final SVM model and file for optimizing the model performance 

**Output
refer to output_snapshot