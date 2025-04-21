
from flask import Flask, render_template, request
import pickle
from pymongo import MongoClient
import os

# Load the saved models
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

# Connect to MongoDB Atlas
MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client['sentimentDB']
collection = db['predictions']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['message']
        transformed_input = tfidf.transform([user_input])
        prediction = clf.predict(transformed_input)[0]

        # Save input and prediction to MongoDB
        data_to_save = {
            'text': user_input,
            'prediction': prediction
        }
        collection.insert_one(data_to_save)

        return render_template('index.html', prediction_text=f'Sentiment: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
