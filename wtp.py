## Scoring Script for Business KPI prediction using Flask
from flask import Flask, jsonify, request
from flask_cors import CORS
#from sklearn.externals import joblib
from sklearn import preprocessing
import pandas as pd
import numpy as np
import json
#import os
#from configparser import ConfigParser
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from werkzeug.exceptions import HTTPException

# initialize flask application
app = Flask(__name__)

###Uncomment if calling from Angular Platform
CORS(app)
#cors = CORS(app, resources={r"/*": {"origins": "*"}})

############################Handling Errors #####################################
@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return jsonify(error=str(e)), code

############################Testing for API #####################################
@app.route('/ccro/msg', methods=['GET'])
def getMessage():
    data = {'message': 'Hello World!'}
    return jsonify(data)

############################Total Water Prediction ###############################
@app.route('/ccro/wtp', methods=['POST'])
def wtp():
    
    # read input json data
    input_query = request.get_json()

    # read reporting frequency
    xin = input_query[0]
    feed = xin.get('Input')
    
    # read prediction horizon
    yin = input_query[1]
    out = yin.get('Output')

    # read raw time series data
    df4 = pd.read_json(json.dumps(input_query[2:]), orient='records')

    # Drop missing values from raw time series data and reset the index of pandas dataframe
    df4 = df4.dropna()
    df4 = df4.reset_index(drop=True)

    # Convert raw time series to differenced (stationary) time series data
    X = df4['TAG025'].diff(1).dropna().values
    train = X
    
    # Train autoregression on differenced time series data
    model = AutoReg(train, lags=1)
    model_fit = model.fit()
    
    # Make prediction as per prediction horizon on differenced data
    differenced = model_fit.predict(start=len(train), end=len(train) + out -1 , dynamic=False).reshape(-1,1)
    
    # Function to invert differenced value
    def inverse_difference(inv_y, x_yorg_minus_1):
        inv = list()
        inv.append(inv_y[0] + x_yorg_minus_1)
        for j in range(len(inv_y)-1):
            value = inv[j] + inv_y[j+1]
            inv.append(value)
        return inv

    # Last non-null value of original time series data
    yorg_minus_1 = df4['TAG025'].iloc[-1:].values
    
    # Inverse differencing of prediction as per raw time series data
    inv_y = np.array(inverse_difference(differenced, yorg_minus_1))

    # Return the last prediction value in json format
    response = json.dumps(inv_y.tolist()[-1])
    return(response)

if __name__ == '__main__':
    # run web server
    app.run(debug=True,host='0.0.0.0',port=8080)
