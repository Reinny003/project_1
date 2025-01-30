from flask import Flask, request, render_template
import numpy as np
import joblib
import openai  # Import OpenAI library

# Load pre-trained models
model_dia = joblib.load("diabetes.joblib")
model_hype = joblib.load("hgbc_model_for_hypertension.joblib")

# Initialize Flask app
app = Flask(__name__, template_folder='templates')  # Corrected __name__

# Base URL
base_url = "https://api.aimlapi.com/v1"
# OpenAI API Keyp
openai.api_key = "c1b1d7df5eef4e5999962faa66ec81b7"

openai.api_base = base_url

# Global variables
global_prediction = None
global_response = None

# Function to generate health recommendations
def generate_health_recommendations(prediction):
    """
    Generate daily health recommendations for managing diabetes and hypertension.
    """
    system_prompt = (
        "You  doctor specializing in  diabetes and hypertension. Provide  daily recommendations "
        "on food, sleep, and exercise to help regulate ."
    )
    user_prompt = f"The patient is {prediction}. Provide specific recommendations for managing their health today."

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=256,
        )
       # Get the raw response from OpenAI
        raw_recommendations = completion.choices[0].message["content"]

        # Clean and format the recommendations
        cleaned_recommendations = (
            raw_recommendations
            .replace("###", "")  # Remove ###
            .replace("\n", "<br>")
            .replace("**","")  # Replace newlines with HTML line breaks
            .strip()  # Remove leading/trailing whitespace
        )

        # Return cleaned recommendations to the template
        return cleaned_recommendations
        

    except Exception as e:
        return f"An error occurred while generating recommendations: {e}"

    
# def format_recommendations(recommendations):
#     # Replace any unexpected newline characters or spacing
#     recommendations = recommendations.replace("\n", "<br>")
#     recommendations = recommendations.replace("##", "<strong>").replace(":", "</strong>")
#     return recommendations


# Home route
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/guidelines')
def guidelines():
    return render_template('guidelines.html',zip=zip)

# Diabetes prediction route
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    global global_prediction, global_response
    prediction = None
    recommendations = ""
    if request.method == 'POST':
        try:
            features = [
                float(request.form['gender']),       # Gender
                float(request.form['age']),          # Age
                float(request.form['urea']),         # Urea
                float(request.form['creatinine']),   # Creatinine
                float(request.form['haemoglobin']),  # Haemoglobin
                float(request.form['chol']),         # Cholesterol
                float(request.form['tg']),           # Triglycerides
                float(request.form['hdl']),          # High-Density Lipoprotein
                float(request.form['ldl']),          # Low-Density Lipoprotein
                float(request.form['vldl']),         # Very-Low-Density Lipoprotein
                float(request.form['bmi']),          # BMI
            ]
            array = np.array(features).reshape(1, -1)
            
            # Make prediction
            pred = model_dia.predict(array)
            
            if pred == 0.0:
                prediction = "High blood sugar"
            elif pred == 1.0:
                prediction = "Low blood sugar"
            else:
                prediction = "Normal blood sugar"
            
            # Update global variables
            global_prediction = prediction
            
            # Generate recommendations
            recommendations = generate_health_recommendations(prediction)
            global_response = recommendations

        except Exception as e:
            return render_template('diabetes.html', error=f"Error: {e}")
        
        return render_template('diabetes.html', prediction=prediction, features=features, recommendations=recommendations)
    
    return render_template('diabetes.html')

# Hypertension prediction route
@app.route('/hypertension', methods=['GET', 'POST'])
def hypertension():
    global global_prediction, global_response
    prediction = None
    recommendations = ""
    if request.method == 'POST':
        try:
            features = [
                float(request.form['male']),
                float(request.form['age']),
                float(request.form['currentSmoker']),
                float(request.form['cigsPerDay']),
                float(request.form['BPMeds']),
                float(request.form['diabetes']),
                float(request.form['totChol']),
                float(request.form['sysBP']),
                float(request.form['diaBP']),
                float(request.form['BMI']),
                float(request.form['heartRate']),
                float(request.form['glucose']),
            ]
            array = np.array(features).reshape(1, -1)
            
            # Make prediction
            pred = model_hype.predict(array)
            
            if pred == 0.0:
                prediction = "High Blood Pressure"
            else:
                prediction = "Low Blood Pressure"
            
            # Update global variables
            global_prediction = prediction
            
            # Generate recommendations
            recommendations = generate_health_recommendations(prediction)
            global_response = recommendations

        except Exception as e:
            return render_template('hypertension.html', error=f"Error: {e}")
        
        return render_template('hypertension.html', prediction=prediction, features=features, recommendations=recommendations,zip=zip)
    
    return render_template('hypertension.html')

if __name__ == "__main__":  # Corrected __name__
    app.run()