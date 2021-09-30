## Scoring Script as Flask
from flask import Flask, jsonify, request
from flask_cors import CORS
#from sklearn.externals import joblib
from sklearn import preprocessing
import pandas as pd
import numpy as np
#import json
#import os
#from configparser import ConfigParser
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA

# initialize flask application
app = Flask(__name__)

###Uncomment if calling from Angular Platform
CORS(app)
#cors = CORS(app, resources={r"/*": {"origins": "*"}})

############################Total Water Prediction ###############################
@app.route('/', methods=['GET'])
def getMessage():
    data = {'message': 'Hello World!'}
    return jsonify(data)

@app.route('/ccro/wtp', methods=['POST'])
def wtp():
    
    # read data
    input_query = request.get_json()
    xin = input_query[0]
    feed = xin.get('Input')
    
    yin = input_query[1]
    out = yin.get('Output')

    df4 = pd.read_json(json.dumps(input_query[2:]), orient='records')
    df4 = df4.dropna()
    df4 = df4.reset_index(drop=True)

    # train autoregression
    X = df4['TAG025'].diff(1).dropna().values
    train = X
    
    model = AutoReg(train, lags=1)
    model_fit = model.fit()
    
    #prediction
    differenced = model_fit.predict(start=len(train), end=len(train) + out -1 , dynamic=False).reshape(-1,1)
    
    # Function to invert differenced value
    def inverse_difference(inv_y, x_yorg_minus_1):
        inv = list()
        inv.append(inv_y[0] + x_yorg_minus_1)
        for j in range(len(inv_y)-1):
            value = inv[j] + inv_y[j+1]
            inv.append(value)
        return inv

    yorg_minus_1 = df4['TAG025'].iloc[-1:].values
    
    # Inverse differencing
    inv_y = np.array(inverse_difference(differenced, yorg_minus_1))
    response = json.dumps(inv_y.tolist())
    return(response)

if __name__ == '__main__':
    # run web server
    app.run(debug=True,host='0.0.0.0',port=80)
