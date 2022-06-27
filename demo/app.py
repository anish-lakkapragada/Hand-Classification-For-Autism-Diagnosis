import tensorflow as tf # machine learning
import cv2  # image processing
import numpy as np # matrix multiplication
import time # record query time
import os 
from flask import Flask, render_template, request, send_from_directory
import json

app = Flask(__name__, static_url_path="")

IMAGE_SIZE = (224, 224, 3)
model = None

def predict_model(model, video): 
    cap = cv2.VideoCapture(video)

    data_matrix = []

    # read in the frames 
    i = 0
    while (cap.isOpened()): 
        ret, frame = cap.read()
        if not ret: break 

        image = cv2.resize(frame, (IMAGE_SIZE[0], IMAGE_SIZE[1])) # resize 
        data_matrix.append(image)
        i += 1

        
    cur_time = time.time()
    prediction = model.predict(np.array([data_matrix]))
    pred_time = time.time() - cur_time
    text_pred = "Hand Flapping" if prediction >= 0.5 else "No Hand Flapping"
    
    cap.release()
    return round(pred_time, 2), (text_pred, round(max([(1 - prediction)[0][0], prediction[0][0]]), 2)) 

@app.before_first_request
def before_first_request():
    global model 
    model = tf.keras.models.load_model("MBNet")


@app.route("/")
def main(): 
    return app.send_static_file('demo.html')

@app.route("/predict", methods=['POST', 'GET'])
def predict(): 
    print(request.get_json())
    file_name = request.get_json()['file']
    if not file_name.find(".mov") and not file_name.find(".mp4"): 
        return "Invalid File!"

    time, (prediction_class, confidence) = predict_model(model, f"static/videos/{file_name}")
    print(time, prediction_class, confidence)
    return {"time": str(time), "prediction": prediction_class, "confidence": str(confidence)}
    

if __name__ == "__main__": 
    app.run(debug=True, port=8000, host="localhost")