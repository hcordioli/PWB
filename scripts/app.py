
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler, MinMaxScaler 
import pandas as pd
import numpy as np

def load_model():
    graph = tf.get_default_graph()
    model = keras.models.load_model('../data/pwb_regression')
    return (graph, model)

def lstm_data_transform(x_data, y_data, num_steps=5):
    """ Changes data to the format for LSTM training for sliding window approach """
    # Prepare the list for the transformed data
    X, y = list(), list()
    # Loop of the entire data set
    for i in range(x_data.shape[0]):
        # compute a new (sliding window) index
        end_ix = i + num_steps
        
        # if index is larger than the size of the dataset, we stop
        if end_ix > x_data.shape[0]:
            break
        
        # Get a sequence of data for x
        seq_X = x_data[i:end_ix]
        
        # Get only the last element of the sequency for y
        seq_y = y_data[end_ix-1]
        
        # Append the list with sequencies
        X.append(seq_X)
        y.append(seq_y)
    
    # Make final arrays
    x_array = np.array(X)
    y_array = np.array(y)
    return x_array, y_array

def predict_pwb(graph, model, pwbds):
    
    FEATURES = [
    'x1', 'x2','x3', 'x4', 'x5', 'x8', 'x9', 'x10', 'x11', 'x13','x14', 'x17', 'x18', 'x19',  'x20', 'x21', 'x26', 'x28', 
    'x30', 'x31','x34', 'x35', 'x37',  'x40', 'x41', 'x43', 'x46','x48', 'x51', 'x53', 'x54', 'x55', 'x57', 'x56','x58', 'x60'
    ]
    
    # Create the dataset with features and filter the data to the list of FEATURES
    pwbds_filtered = pwbds[FEATURES]
    # Get the number of rows in the data
    nrows = pwbds_filtered.shape[0]

    # Convert the data to numpy values
    np_data_unscaled = np.array(pwbds_filtered)
    np_data = np.reshape(np_data_unscaled, (nrows, -1))

    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(np_data_unscaled)
    pwbds_scaled = pd.DataFrame(
        np_data_scaled,
        columns=FEATURES
    )
#    yds = pwbds_scaled.pop('y')
    yds = np.zeros(pwbds_scaled.shape[0])
    num_steps = 5 # Intervalo entre medições é de 2 minutos
    x_new, y_new = lstm_data_transform(pwbds_scaled, yds, num_steps=num_steps)
    with graph.as_default():
        test_predict = model.predict(x_new)   
    return test_predict

# We must now create our app, which I will store in a variable called app:
app = Flask(__name__)

@app.route("/signals", methods=["POST"])
def predict_pwbds():
    graph, model = load_model()
    print(model.summary())
    if request.is_json:
        signals = request.get_json()
        signalsdf = pd.DataFrame(signals)
        y_pred = predict_pwb(graph, model, signalsdf)
        predicts = (y_pred.tolist())
        makeitastring = ''.join(map(str, predicts))
        return (makeitastring, 201)
    return {"error": "Request must be JSON"}, 415