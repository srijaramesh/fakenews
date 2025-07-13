from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        news = request.form['news']
        input_vec = vectorizer.transform([news])
        prediction = model.predict(input_vec)[0]
        result = "Real News ✅" if prediction == 1 else "Fake News ❌"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
