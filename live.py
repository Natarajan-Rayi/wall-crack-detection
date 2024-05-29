from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
from engineio.async_drivers import gevent


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

def detect_cracks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:  # Adjusted the area threshold to filter out smaller cracks
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            
            if aspect_ratio < 5 and solidity > 0.5:  # Filtering based on aspect ratio and solidity
                filtered_contours.append(contour)
    
    return filtered_contours

def draw_cracks(image, contours):
    for contour in contours:
        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
    
    if contours:
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        text_x = x + w + 5
        text_y = y + 15
        cv2.rectangle(image, (text_x, y), (text_x + 60, y + 20), (0, 0, 255), -1)
        cv2.putText(image, 'crack', (text_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('stream')
def handle_stream(data):
    try:
        print("Received frame for processing")
        # Decode the incoming image
        image_data = base64.b64decode(data.split(',')[1])
        image = np.array(Image.open(io.BytesIO(image_data)))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        print("Detecting cracks")
        # Detect cracks
        contours = detect_cracks(image)
        if not contours:
            print("No cracks detected")
        else:
            print(f"Detected {len(contours)} cracks")

        image_with_cracks = draw_cracks(image, contours)
        
        # Encode the image with cracks back to base64
        _, buffer = cv2.imencode('.jpg', image_with_cracks)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        emit('response', image_base64)
        print("Frame processed and sent back")
    except Exception as e:
        print(f"Error processing frame: {e}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
