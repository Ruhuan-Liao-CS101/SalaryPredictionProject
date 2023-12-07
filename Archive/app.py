import flask
import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import mean_absolute_error, mean_absolute_error, r2_score

app = Flask(__name__)


def init():
    print("initializing... ")


# Load the trained model
model = joblib.load(
    '/Users/ruhuanliao/Fall 2023/AI/PredictProject/Archive/SalaryPrediction.pkl')


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = float(request.form['age'])
        gender = request.form['gender']
        education_level = request.form['education_level']
        job_title = request.form['job_title']
        years_of_experience = float(request.form['years_of_experience'])

        # Make a prediction using the loaded model
        input_data = {
            'Age': age,
            'Gender': map_gender(gender),
            'Education Level': map_education(education_level),
            'Job Title': int(job_title),
            'Years of Experience': years_of_experience
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(
            input_df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']])

        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return f"Error: {e}"


def map_gender(gender):
    gender_mapping = {'female': 0, 'male': 1, 'other': 2}
    return gender_mapping.get(gender.lower(), 0)


def map_education(education):
    education_mapping = {"bachelor": 0,
                         "bachelor's Degree": 1,
                         'high_school': 2,
                         "master": 3,
                         "master's Degree": 4,
                         'PhD': 5,
                         'Unknown': 6,
                         'phD': 7}
    return education_mapping.get(education.lower(), 4)


if __name__ == '__main__':
    init()
    app.run(debug=True, port=9090, use_reloader=True)
