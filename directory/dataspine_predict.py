import os
import numpy as np
import json
import logging

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['predict']


_labels= {'model_runtime': os.environ['DATASPINE_MODEL_RUNTIME'],
          'model_type': os.environ['DATASPINE_MODEL_TYPE'],
          'model_name': os.environ['DATASPINE_MODEL_NAME'],
          'model_tag': os.environ['DATASPINE_MODEL_TAG']}


def _initialize_upon_import() -> TensorFlowServingModel:
    ''' Initialize / Restore Model Object.
    '''
    pass
    

# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def predict(request: bytes) -> bytes:
    '''Where the magic happens...'''

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="predict"):
        predictions = _model.predict(transformed_request)

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(predictions)

    return transformed_response


def _transform_request(request: bytes) -> dict:
#    request_str = request.decode('utf-8')
#    request_json = json.loads(request_str)
#    request_np = ((255 - np.array(request_json['image'], dtype=np.uint8)) / 255.0).reshape(1, 28, 28)
#    return {'image': request_np}
    pass


def _transform_response(response: dict) -> json:
    #return json.dumps({'classes': response['classes'].tolist(), 'probabilities': response['probabilities'].tolist()})
    pass