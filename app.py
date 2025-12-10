from flask import Flask, request, render_template, flash, redirect, url_for
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application
app.secret_key = "replace-with-a-secure-random-key"  # needed for flash messages

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    # POST handling
    try:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
    except (TypeError, ValueError) as ex:
        flash("Please make sure numeric fields (reading/writing scores) are filled with valid numbers.")
        return redirect(url_for('predict_datapoint'))

    pred_df = data.get_data_as_dataframe()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline = PredictPipeline()
    print("Mid Prediction")
    results = predict_pipeline.predict(pred_df)
    print("After Prediction:", results)

    # assume results is an array-like; show first element
    predicted_value = results[0] if hasattr(results, "__len__") else results
    return render_template('home.html', results=predicted_value)

if __name__ == "__main__":
    # For development only: set debug=True. Remove in production.
    app.run(host="0.0.0.0", port=5000, debug=True)
