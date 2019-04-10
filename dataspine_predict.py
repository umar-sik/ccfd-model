import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TENSORFLOW_FLAGS'] = 'floatX=float32,device=cpu'

import pandas as pd
import numpy as np
import json
import logging
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras_tf_model import KerasTensorflowModel
from keras.models import Model, load_model
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

_logger = logging.getLogger('model_logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['predict']


_labels= {'model_runtime': os.environ['DATASPINE_MODEL_RUNTIME'],
          'model_type': os.environ['DATASPINE_MODEL_TYPE'],
          'model_name': os.environ['DATASPINE_MODEL_NAME'],
          'model_tag': os.environ['DATASPINE_MODEL_TAG']}


def _initialize_upon_import(model_state_path: str) -> KerasTensorflowModel:
    ''' Initialize / Restore Model Object.
    '''
    return KerasTensorflowModel(model_state_path)


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credit_card_fraud_model.h5'))

@monitor(labels=_labels, name="transform_request")
def _json_to_pd_df(request: bytes) -> np.array:
    request_str = request.decode('utf-8')
    #request_str = request_str.strip().replace('\n', ',')
    # surround the json with '[' ']' to prepare for conversion
    #request_str = '[%s]' % request_str
    #request_json = json.loads(request_str)
    df_data = pd.read_json(request_str,orient='records')
    return (df_data)


@monitor(labels=_labels, name="transform_response")
def _numpy_to_json(response: np.array) -> bytes:
    return pd.Series(list(response)).to_json(orient='split')


@log(labels=_labels, logger=_logger)
def predict(request: bytes) -> bytes:
    '''Where the magic happens...'''
    transformed_request = _json_to_pd_df(request)
    data =  transformed_request.drop('Time', axis=1)
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    y_test = data['Class']
    X_test = data.drop(['Class'], axis=1)

    X_values = X_test.values
    #autoencoder = load_model(_model)
    
    with monitor(labels=_labels, name="predict"):
        predictions = _model.predict(X_test)
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': y_test})
    # to build the confusion matrix
    threshold = 2.9

    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

    conf_matrix = confusion_matrix(error_df.true_class, y_pred)

    return _numpy_to_json(conf_matrix)
