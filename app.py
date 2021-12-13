#import Flask 
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import pickle
import theano

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def Weather_Predict():
    return render_template('home.html')

@app.route('/predict/', methods=['POST','GET'])
def getprediction():
    inputs = [float(x) for x in request.form.values()]
    final_input = [np.array(input)]
    prediction = model(inputs,[0, 0, 0, 0, 0, 0, 0, 1])
    output = prediction[0]

    return render_template('/predict.html', output= 'Expected Weather Condition Is :{}'.format(output), prediction=prediction)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)