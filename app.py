from src.InsurencepPricePrediction.pipelines.prediction_pipeline import custom_data,model_prediction
from flask import Flask,request,render_template
import numpy as np
app=Flask(__name__,template_folder="template")
@app.route('/')
def home_page():
    return render_template("form.html")
@app.route('/predict',methods=["POST"])
def pred_page():   
        get_data=custom_data(age=float(request.form.get('age')),
                             sex=request.form.get('sex'),
                             bmi=float(request.form.get('bmi')),
                             children=float(request.form.get('children')),
                             smoker=request.form.get('smoker'),
                             region=request.form.get('region'))
        final_data=get_data.get_data_as_dataframe()
        pred=model_prediction()
        x=pred.model_pred_initiate(final_data)
        return render_template("result.html",x=x)

if __name__=="__main__":
 app.run()