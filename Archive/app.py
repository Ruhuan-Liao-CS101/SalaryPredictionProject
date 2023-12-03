import flask
import joblib
import pandas as pd
from flask import Flask, render_template, request

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
            'Gender': gender,
            'Education Level': map_education(education_level),
            'Job Title': map_job_title(job_title),
            'Years of Experience': years_of_experience
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(
            input_df[['Age', 'Years of Experience', 'Education Level', 'Job Title', 'Years of Experience']])

        # Render the result page with the prediction
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return f"Error: {e}"


def map_gender(gender):
    gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
    return gender_mapping.get(gender, 0)


def map_education(education):
    education_mapping = {"Bachelor's": 0,
                         "Bachelor's Degree": 1,
                         'High School': 2,
                         "Master's": 3,
                         "Master's Degree": 4,
                         'PhD': 5,
                         'Unknown': 6,
                         'phD': 7}
    return education_mapping.get(education.lower(), 4)


def map_job_title(job_title):
    job_title_mapping = {'Account Manager': 0,
                         'Accountant': 1,
                         'Administrative Assistant': 2,
                         'Back end Developer': 3,
                         'Business Analyst': 4,
                         'Business Development Manager': 5,
                         'Business Intelligence Analyst': 6,
                         'CEO': 7,
                         'Chief Data Officer': 8,
                         'Chief Technology Officer': 9,
                         'Content Marketing Manager': 10,
                         'Copywriter': 11,
                         'Creative Director': 12,
                         'Customer Service Manager': 13,
                         'Customer Service Rep': 14,
                         'Customer Service Representative': 15,
                         'Customer Success Manager': 16,
                         'Customer Success Rep': 17,
                         'Data Analyst': 18,
                         'Data Entry Clerk': 19,
                         'Data Scientist': 20,
                         'Delivery Driver': 21,
                         'Developer': 22,
                         'Digital Content Producer': 23,
                         'Digital Marketing Manager': 24,
                         'Digital Marketing Specialist': 25,
                         'Director': 26,
                         'Director of Business Development': 27,
                         'Director of Data Science': 28,
                         'Director of Engineering': 29,
                         'Director of Finance': 30,
                         'Director of HR': 31,
                         'Director of Human Capital': 32,
                         'Director of Human Resources': 33,
                         'Director of Marketing': 34,
                         'Director of Operations': 35,
                         'Director of Product Management': 36,
                         'Director of Sales': 37,
                         'Director of Sales and Marketing': 38,
                         'Event Coordinator': 39,
                         'Financial Advisor': 40,
                         'Financial Analyst': 41,
                         'Financial Manager': 42,
                         'Front End Developer': 43,
                         'Front end Developer': 44,
                         'Full Stack Engineer': 45,
                         'Graphic Designer': 46,
                         'HR Generalist': 47,
                         'HR Manager': 48,
                         'Help Desk Analyst': 49,
                         'Human Resources Coordinator': 50,
                         'Human Resources Director': 51,
                         'Human Resources Manager': 52,
                         'IT Manager': 53,
                         'IT Support': 54,
                         'IT Support Specialist': 55,
                         'Junior Account Manager': 56,
                         'Junior Accountant': 57,
                         'Junior Advertising Coordinator': 58,
                         'Junior Business Analyst': 59,
                         'Junior Business Development Associate': 60,
                         'Junior Business Operations Analyst': 61,
                         'Junior Copywriter': 62,
                         'Junior Customer Support Specialist': 63,
                         'Junior Data Analyst': 64,
                         'Junior Data Scientist': 65,
                         'Junior Designer': 66,
                         'Junior Developer': 67,
                         'Junior Financial Advisor': 68,
                         'Junior Financial Analyst': 69,
                         'Junior HR Coordinator': 70,
                         'Junior HR Generalist': 71,
                         'Junior Marketing Analyst': 72,
                         'Junior Marketing Coordinator': 73,
                         'Junior Marketing Manager': 74,
                         'Junior Marketing Specialist': 75,
                         'Junior Operations Analyst': 76,
                         'Junior Operations Coordinator': 77,
                         'Junior Operations Manager': 78,
                         'Junior Product Manager': 79,
                         'Junior Project Manager': 80,
                         'Junior Recruiter': 81,
                         'Junior Research Scientist': 82,
                         'Junior Sales Associate': 83,
                         'Junior Sales Representative': 84,
                         'Junior Social Media Manager': 85,
                         'Junior Social Media Specialist': 86,
                         'Junior Software Developer': 87,
                         'Junior Software Engineer': 88,
                         'Junior UX Designer': 89,
                         'Junior Web Designer': 90,
                         'Junior Web Developer': 91,
                         'Juniour HR Coordinator': 92,
                         'Juniour HR Generalist': 93,
                         'Marketing Analyst': 94,
                         'Marketing Coordinator': 95,
                         'Marketing Director': 96,
                         'Marketing Manager': 97,
                         'Marketing Specialist': 98,
                         'Network Engineer': 99,
                         'Office Manager': 100,
                         'Operations Analyst': 101,
                         'Operations Director': 102,
                         'Operations Manager': 103,
                         'Principal Engineer': 104,
                         'Principal Scientist': 105,
                         'Product Designer': 106,
                         'Product Manager': 107,
                         'Product Marketing Manager': 108,
                         'Project Engineer': 109,
                         'Project Manager': 110,
                         'Public Relations Manager': 111,
                         'Receptionist': 112,
                         'Recruiter': 113,
                         'Research Director': 114,
                         'Research Scientist': 115,
                         'Sales Associate': 116,
                         'Sales Director': 117,
                         'Sales Executive': 118,
                         'Sales Manager': 119,
                         'Sales Operations Manager': 120,
                         'Sales Representative': 121,
                         'Senior Account Executive': 122,
                         'Senior Account Manager': 123,
                         'Senior Accountant': 124,
                         'Senior Business Analyst': 125,
                         'Senior Business Development Manager': 126,
                         'Senior Consultant': 127,
                         'Senior Data Analyst': 128,
                         'Senior Data Engineer': 129,
                         'Senior Data Scientist': 130,
                         'Senior Engineer': 131,
                         'Senior Financial Advisor': 132,
                         'Senior Financial Analyst': 133,
                         'Senior Financial Manager': 134,
                         'Senior Graphic Designer': 135,
                         'Senior HR Generalist': 136,
                         'Senior HR Manager': 137,
                         'Senior HR Specialist': 138,
                         'Senior Human Resources Coordinator': 139,
                         'Senior Human Resources Manager': 140,
                         'Senior Human Resources Specialist': 141,
                         'Senior IT Consultant': 142,
                         'Senior IT Project Manager': 143,
                         'Senior IT Support Specialist': 144,
                         'Senior Manager': 145,
                         'Senior Marketing Analyst': 146,
                         'Senior Marketing Coordinator': 147,
                         'Senior Marketing Director': 148,
                         'Senior Marketing Manager': 149,
                         'Senior Marketing Specialist': 150,
                         'Senior Operations Analyst': 151,
                         'Senior Operations Coordinator': 152,
                         'Senior Operations Manager': 153,
                         'Senior Product Designer': 154,
                         'Senior Product Development Manager': 155,
                         'Senior Product Manager': 156,
                         'Senior Product Marketing Manager': 157,
                         'Senior Project Coordinator': 158,
                         'Senior Project Engineer': 159,
                         'Senior Project Manager': 160,
                         'Senior Quality Assurance Analyst': 161,
                         'Senior Research Scientist': 162,
                         'Senior Researcher': 163,
                         'Senior Sales Manager': 164,
                         'Senior Sales Representative': 165,
                         'Senior Scientist': 166,
                         'Senior Software Architect': 167,
                         'Senior Software Developer': 168,
                         'Senior Software Engineer': 169,
                         'Senior Training Specialist': 170,
                         'Senior UX Designer': 171,
                         'Social M': 172,
                         'Social Media Man': 173,
                         'Social Media Manager': 174,
                         'Social Media Specialist': 175,
                         'Software Developer': 176,
                         'Software Engineer': 177,
                         'Software Engineer Manager': 178,
                         'Software Manager': 179,
                         'Software Project Manager': 180,
                         'Strategy Consultant': 181,
                         'Supply Chain Analyst': 182,
                         'Supply Chain Manager': 183,
                         'Technical Recruiter': 184,
                         'Technical Support Specialist': 185,
                         'Technical Writer': 186,
                         'Training Specialist': 187,
                         'UX Designer': 188,
                         'UX Researcher': 189,
                         'Unknown': 190,
                         'VP of Finance': 191,
                         'VP of Operations': 192,
                         'Web Developer': 193}
    return job_title_mapping.get(job_title, 0)


if __name__ == '__main__':
    init()
    app.run(debug=True, port=9090)
