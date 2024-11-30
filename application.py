from flask import Flask,request,render_template,app,Response
import numpy as np
import pandas as np
import pickle

application = Flask(__name__)
app = application

scaler = pickle.load(open("Model/standardscaler.pkl","rb"))
classifier = pickle.load(open("Model/classifier.pkl","rb"))

## Home Route

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def pridect():
    result = ""
    
    if request.method=="POST":
        Pregnancies=int(request.form.get("pregnancies"))
        Glucose = float(request.form.get('glucose'))
        BloodPressure = float(request.form.get('bloodpressure'))
        SkinThickness = float(request.form.get('skinthickness'))
        Insulin = float(request.form.get('insulin'))
        BMI = float(request.form.get('bmi'))
        DiabetesPedigreeFunction = float(request.form.get('diabetespedigree'))
        Age = float(request.form.get('age'))
        
        new_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness, Insulin,BMI, DiabetesPedigreeFunction,Age]])
        predict = classifier.predict(new_data)
        
        if predict[0]==1:
            result = "Diabetic"
        
        else:
            result = "Non Diabetic"
            
    return render_template("index.html",result = result)

if __name__=="__main__":
    app.run(host="0.0.0.0")
