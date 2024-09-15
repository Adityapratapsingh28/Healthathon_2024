from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    prediction_proba = model.predict_proba(final_features)

    output = prediction[0]
    confidence = prediction_proba[0][1] if output == 1 else prediction_proba[0][0]

    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()
    top_features = [feature_names[i] for i in sorted_idx[-3:]]

    result = "Diabetic" if output == 1 else "Not Diabetic"
    additional_info = ""

    if output == 1:
        if 'Pregnancies' in top_features and features[0] > 0:
            additional_info = "Consider screening for Gestational Diabetes."
        elif 'Age' in top_features and features[7] < 40:
            additional_info = "Type 1 Diabetes might be more likely."
        else:
            additional_info = "Type 2 Diabetes is the most probable, but further medical tests are needed for confirmation."

    return render_template('result.html', prediction=result, confidence=confidence, 
                           top_features=top_features, additional_info=additional_info)

if __name__ == "__main__":
    app.run(debug=True)