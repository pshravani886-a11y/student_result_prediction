from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("result_prediction_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict_page')
def predict_page():
    return render_template("predict.html")


@app.route('/predict', methods=['POST'])
def predict():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    parental_education=int(request.form['parental_education'])
    internet_access=int(request.form['internet_access'])
    extracurricular_participation=int(request.form['extracurricular_participation'])
    final_exam_score=int(request.form['final_exam_score'])

    study_hours = float(request.form['study_hours'])
    attendance = float(request.form['attendance'])
    previous_grade = float(request.form['previous_grade'])
    assignments = int(request.form['assignments'])
    sleep_hours = float(request.form['sleep_hours'])

    features = np.array([[gender,age,study_hours, attendance, previous_grade, assignments,parental_education,internet_access,extracurricular_participation,final_exam_score, sleep_hours]])

    prediction = model.predict(features)

    output = round(prediction[0],2)

    return render_template("predict.html", prediction_text=f"Predicted Exam Score: {output}")


if __name__ == "__main__":
    app.run(debug=True)