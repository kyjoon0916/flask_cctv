from flask import Flask, render_template, Response
import cv2
import time
import json
import numpy as np


with open("passwd.json","r") as passwd:
    pwd = json.load(passwd)
    
source = pwd["source"] 
cap = cv2.VideoCapture(source)

app = Flask(__name__)
def gen_frames():
    while True:
 #   frame frame loop read the data of the camera
        success, frame = cap.read()
        if not success:
            break
        else:
 # Code data of each frame and store it in Memory
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
 # Use the yield statement to return the frame data as the responder, Content-Type is image / jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
@app.route('/rtsp')
def video_start():
 # By returning an image of a frame of frame, it reaches the purpose of watching the video. Multipart / X-Mixed-Replace is a single HTTP request - response mode, if the network is interrupted, it will cause the video stream to terminate, you must reconnect to recover
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/')
def index():
    return render_template('index.html')
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
      