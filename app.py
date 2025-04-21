from flask import Flask, render_template, request
import pickle
from pymongo import MongoClient
import os

# Load the saved models
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

# Connect to MongoDB Atlas using environment variable
MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client['sentimentDB']
collection = db['predictions']

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    sentiment = None
    message = ""

    if request.method == 'POST':
        message = request.form['message']
        transformed_input = tfidf.transform([message])
        prediction = clf.predict(transformed_input)[0]

        # Save the prediction to MongoDB
        collection.insert_one({
            'text': message,
            'prediction': int(prediction)
        })

        sentiment = int(prediction)

    return render_template('index.html', sentiment=sentiment, comment=message)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
