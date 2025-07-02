from flask import Flask ,request,jsonify
import pandas as pd
import pickle
from flask_cors import CORS
import numpy as np
app=Flask(__name__)
CORS(app)
Dmodel=pickle.load(open("models/diabeties_Rf.pkl","rb"))
Dscaler=pickle.load(open("models/diabeties_scaler.pkl","rb"))
Hmodel=pickle.load(open("models/RFheart.pkl","rb"))
Hscaler=pickle.load(open("models/scalerheart.pkl","rb"))
@app.route("/getdiabitiesprediction",methods=["GET","POST"])
def prediction():
    if request.method=="POST":
        data = request.get_json()
        
        features=[
            data.get("Pregnancies"),
            data.get("Glucose"),
            data.get("Insulin"),
            data.get("BMI"),
            data.get("Diabetes Pedigree Function"),
            data.get("Age"),
        ]
        
        inputs=np.array([features])
        scaled_input=Dscaler.transform(inputs)
        ypred=Dmodel.predict(scaled_input)
        
        return jsonify({"message": "Success", "prediction": int(ypred[0])})




@app.route("/getheartprediction",methods=["GET","POST"])
def predictionh():
    if request.method=="POST":
        data = request.get_json()
        
        features=[
            data.get("Age"),
            data.get("Sex"),
            data.get("Chest Pain Type"),
            data.get("Resting BP"),
            data.get("Max Heart Rate"),
            data.get("Exercise Induced Angina"),
            data.get("Oldpeak"),
            data.get("Slope of ST segment"),
            data.get("Major Vessels"),
            data.get("Thalassemia"),
        ]
        
        inputs=np.array([features])
        scaled_input=Hscaler.transform(inputs)
        ypred=Hmodel.predict(scaled_input)
        
        if int(ypred[0])==1:
            final=0 
        else:
            final=1    
        return jsonify({"message": "Success", "prediction": final})



if __name__ =="__main__":
    app.run(host="0.0.0.0")