pip install scikit==1.4.2
from flask import Flask,request,render_template
from joblib import load
import numpy as np
app=Flask(__name__)

with open('model .joblib','rb') as model_file:
    model=load(model_file)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        age = request.form.get('age', '')  # Get 'age' value or default to empty string
        gender= request.form.get('gender', '')
        salary = request.form.get('salary', '')  # Get 'estimated salary' value or default to empty string

    # Validate input
    if not all([age,gender,salary]):
        # Handle case where required fields are empty
        return render_template('index.html', pred_res='Please fill out all fields.')

    try:
        age = int(age)
        gender= int(gender)
        salary = int(salary)
    except ValueError:
        # Handle case where non-numeric input is provided
        return render_template('index.html', pred_res='Invalid input. Age must be an integer and estimated salary must be a number.')

    feature = np.array([[age,gender,salary]]) 
    prediction = model.predict(feature)

    pred_res="Will Purchase" if prediction[0]==1 else "Will not Purchase"

    return render_template('index.html', pred_res=pred_res)
    

if __name__=='__main__':
    app.run(debug=True)
    
