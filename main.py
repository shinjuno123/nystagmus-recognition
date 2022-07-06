from flask import Flask, request
from flask_restx import Api, Resource
import tensorflow as tf
import json
import numpy as np


app = Flask(__name__)
api = Api(app)

def getModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96,return_sequences=True), input_shape=(15, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1024,activation = "relu"))
    model.add(tf.keras.layers.Dense(512,activation = "relu"))
    model.add(tf.keras.layers.Dense(128,activation = "relu"))
    model.add(tf.keras.layers.Dense(5,activation = "softmax"))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0004)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
    model.summary()
    return model

class NumpyEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj,np.ndarray):
            return obj.tolist()
        return json.JSONDecoder.default(self,obj)

@api.route('/',methods=['GET','POST'])
class TFProcess(Resource):
    def get(self):
        return {'description':"This is temperature web server for Tensorflow"}
    def post(self):
        content_type = request.headers.get('Content-Type')
        if content_type == 'application/json':
            model = getModel()
            model.load_weights("./weight/Classifier.h5")
            json_load = json.loads(request.json)
            json_restored = np.asarray(json_load['sliced_point'])
            result = model.predict(json_restored)
            json_dump = json.dumps({'result':result},cls=NumpyEncoder)
            return json_dump
        return {'description':"check if your body is json format"}
    
if __name__== "__main__":
    app.run(debug=True,host='0.0.0.0',port=8080)