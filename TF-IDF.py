# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import pickle

# file_path = r'D:/Research/dataset/sample/sample.csv'
# data = pd.read_csv(file_path, encoding='latin-1')

# # Correct column name based on the dataset
# column_name = 'Question'

# questions = data[column_name].values.astype('U')

# # Create and train the TF-IDF vectorizer
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)  # Ensure max_features matches model
# tfidf_vectorizer.fit(questions)

# # Save the vectorizer to a file
# with open('model/tfidf_vectorizer_trained.pkl', 'wb') as vectorizer_file:
#     pickle.dump(tfidf_vectorizer, vectorizer_file)

# # Train and save the model (if not already done)
# from sklearn.svm import SVC

# # Assuming 'Category' column contains the labels
# categories = data['Category']

# model = SVC()
# model.fit(tfidf_vectorizer.transform(questions), categories)

# with open('model/best_model.pkl', 'wb') as model_file:
#     pickle.dump(model, model_file)

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd

file_path = r'D:/Research/dataset/sample/sample.csv'
data = pd.read_csv(file_path, encoding='latin-1')

# Correct column name based on the dataset
column_name = 'Question'
questions = data[column_name].values.astype('U')

# Create and train the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=15000)
tfidf_vectorizer.fit(questions)

# Save the vectorizer to a file
with open('model/tfidf_vectorizer_trained.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)
