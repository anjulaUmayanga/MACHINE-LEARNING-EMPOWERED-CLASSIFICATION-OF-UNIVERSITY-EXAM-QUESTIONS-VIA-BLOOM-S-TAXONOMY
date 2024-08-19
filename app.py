# # from flask import Flask, request, render_template
# # import pickle
# # from sklearn.feature_extraction.text import TfidfVectorizer

# # app = Flask(__name__)

# # # Load the pre-trained model and vectorizer
# # with open('model/best_model.pkl', 'rb') as model_file:
# #     best_model = pickle.load(model_file)

# # with open('model/tfidf_vectorizer_trained.pkl', 'rb') as vectorizer_file:
# #     tfidf_vectorizer = pickle.load(vectorizer_file)

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/classify', methods=['POST'])
# # def classify():
# #     num_questions = int(request.form['num_questions'])
# #     questions = [request.form[f'question{i+1}'] for i in range(num_questions)]

# #     # Preprocess the questions
# #     processed_questions = preprocess_questions(questions)

# #     # Vectorize the questions
# #     vectors = tfidf_vectorizer.transform(processed_questions)

# #     # Predict the categories
# #     predictions = best_model.predict(vectors)

# #     results = zip(questions, predictions)

# #     return render_template('result.html', results=results)

# # def preprocess_questions(questions):
# #     # Placeholder for preprocessing function, can be replaced with actual preprocessing steps
# #     return questions

# # if __name__ == '__main__':
# #     app.run(debug=True)


# from flask import Flask, request, render_template
# import pickle
# import numpy as np
# from collections import Counter

# app = Flask(__name__)

# # Load the pre-trained model and vectorizer
# with open('model/best_model.pkl', 'rb') as model_file:
#     best_model = pickle.load(model_file)

# with open('model/tfidf_vectorizer_trained.pkl', 'rb') as vectorizer_file:
#     tfidf_vectorizer = pickle.load(vectorizer_file)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/classify', methods=['POST'])
# def classify():
#     num_questions = int(request.form['num_questions'])
#     questions = [request.form[f'question{i+1}'] for i in range(num_questions)]

#     # Preprocess the questions
#     processed_questions = preprocess_questions(questions)

#     # Vectorize the questions
#     vectors = tfidf_vectorizer.transform(processed_questions)

#     # Predict the categories
#     predictions = best_model.predict(vectors)

#     results = list(zip(questions, predictions))

#     # Calculate category percentages
#     total_questions = len(predictions)
#     category_counts = Counter(predictions)
#     category_percentages = {category: (count / total_questions) * 100 for category, count in category_counts.items()}

#     return render_template('result.html', results=results, total_questions=total_questions, category_percentages=category_percentages)

# def preprocess_questions(questions):
#     # Placeholder for preprocessing function, can be replaced with actual preprocessing steps
#     return questions

# if __name__ == '__main__':
#     app.run(debug=True)
#######################################################
# from flask import Flask, request, render_template
# import pickle
# from collections import Counter

# app = Flask(__name__)

# # Load the pre-trained model and vectorizer
# try:
#     with open('model/best_model.pkl', 'rb') as model_file:
#         best_model = pickle.load(model_file)
#     with open('model/tfidf_vectorizer_trained.pkl', 'rb') as vectorizer_file:
#         tfidf_vectorizer = pickle.load(vectorizer_file)
# except FileNotFoundError as e:
#     print(f"File not found: {e}")
#     exit()
# except pickle.UnpicklingError as e:
#     print(f"Error unpickling file: {e}")
#     exit()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/classify', methods=['POST'])
# def classify():
#     num_questions = int(request.form['num_questions'])
#     questions = [request.form[f'question{i+1}'] for i in range(num_questions)]

#     # Preprocess the questions
#     processed_questions = preprocess_questions(questions)

#     # Check if tfidf_vectorizer is fitted
#     if not hasattr(tfidf_vectorizer, 'idf_'):
#         print("TF-IDF Vectorizer is not fitted.")
#         return "TF-IDF Vectorizer is not fitted."

#     # Vectorize the questions
#     try:
#         vectors = tfidf_vectorizer.transform(processed_questions)
#     except Exception as e:
#         print(f"Error during vectorization: {e}")
#         return "Error during vectorization."

#     # Predict the categories
#     try:
#         predictions = best_model.predict(vectors)
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         return "Error during prediction."

#     results = list(zip(questions, predictions))

#     # Calculate category percentages
#     total_questions = len(predictions)
#     category_counts = Counter(predictions)
#     category_percentages = {category: (count / total_questions) * 100 for category, count in category_counts.items()}

#     return render_template('result.html', results=results, total_questions=total_questions, category_percentages=category_percentages)

# def preprocess_questions(questions):
#     # Placeholder for preprocessing function, can be replaced with actual preprocessing steps
#     return questions

# if __name__ == '__main__':
#     app.run(debug=True)

#########################
# from flask import Flask, request, render_template
# import pickle
# import numpy as np
# from collections import Counter

# app = Flask(__name__)

# # Load the pre-trained model and vectorizer
# with open('model/best_model.pkl', 'rb') as model_file:
#     best_model = pickle.load(model_file)

# with open('model/tfidf_vectorizer_trained.pkl', 'rb') as vectorizer_file:
#     tfidf_vectorizer = pickle.load(vectorizer_file)

# @app.route('/')
# def welcome():
#     return render_template('welcome.html')

# @app.route('/index')
# def index():
#     return render_template  ('index.html')

# @app.route('/classify', methods=['POST'])
# def classify():
#     num_questions = int(request.form['num_questions'])
#     questions = [request.form[f'question{i+1}'] for i in range(num_questions)]

#     # Preprocess the questions
#     processed_questions = preprocess_questions(questions)

#     # Vectorize the questions
#     vectors = tfidf_vectorizer.transform(processed_questions)

#     # Predict the categories
#     predictions = best_model.predict(vectors)

#     results = list(zip(questions, predictions))

#     # Calculate category percentages
#     total_questions = len(predictions)
#     category_counts = Counter(predictions)
#     category_percentages = {category: round((count / total_questions) * 100, 2) for category, count in category_counts.items()}

#     return render_template('result.html', results=results, total_questions=total_questions, category_percentages=category_percentages)

# def preprocess_questions(questions):
#     # Placeholder for preprocessing function, can be replaced with actual preprocessing steps
#     return questions

# if __name__ == '__main__':
#     app.run(debug=True)

# ###################
from flask import Flask, request, render_template
import pickle
import pandas as pd
from collections import Counter

app = Flask(__name__)

# Load the pre-trained model and vectorizer (ensure these are the same as used during training)
with open('model/best_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

with open('model/tfidf_vectorizer_trained.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    num_questions = int(request.form['num_questions'])
    questions = [request.form[f'question{i+1}'] for i in range(num_questions)]

    # Preprocess the questions (must be the same as used during model training)
    processed_questions = preprocess_questions(questions)

    # Vectorize the questions using the same vectorizer
    vectors = tfidf_vectorizer.transform(processed_questions)

    # Predict the categories
    predictions = best_model.predict(vectors)

    results = list(zip(questions, predictions))

    # Calculate category percentages
    total_questions = len(predictions)
    category_counts = Counter(predictions)
    category_percentages = {category: round((count / total_questions) * 100, 2) for category, count in category_counts.items()}

    return render_template('result.html', results=results, total_questions=total_questions, category_percentages=category_percentages)

def preprocess_questions(questions):
    # Placeholder for preprocessing function, make sure it matches the training phase
    return questions

if __name__ == '__main__':
    app.run(debug=True)

