from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words("english"))

responses = {
    "Order Status": "You can track your order from the Orders section.",
    "Returns & Refunds": "You can request a return or refund from your account.",
    "Billing & Payments": "Please check your payment details or contact billing support.",
    "Account Issues": "Try resetting your password or verifying your account.",
    "Technical Support": "Our technical team will assist you shortly.",
    "General Inquiry": "Thank you for contacting us. How can we help you?"
}

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    category = ""
    response = ""
    if request.method == "POST":
        user_query = request.form["query"]
        clean_query = preprocess(user_query)
        vector = vectorizer.transform([clean_query])
        category = model.predict(vector)[0]
        response = responses.get(category)

    return render_template("index.html", category=category, response=response)

if __name__ == "__main__":
    app.run(debug=True)
