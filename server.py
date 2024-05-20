from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

crack_count = 0

def perform_crack_detection(frame):
    global crack_count

    # Reset crack count for each frame
    crack_count = 0

    # Calculate total area of the frame
    total_area = frame.shape[0] * frame.shape[1]

    # Crack detection code
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to identify wall cracks
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter out small contours
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 50:  # Filter out contours with small perimeter
                # Filter out contours not located near the edges
                x, y, w, h = cv2.boundingRect(contour)
                if x > 10 and y > 10 and x + w < frame.shape[1] - 10 and y + h < frame.shape[0] - 10:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
                    crack_count += 1

                    # Draw rectangle around the detected crack
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Calculate percentage of crack area
                    crack_area_percentage = (area / total_area) * 100

                    # Display percentage on the frame
                    cv2.putText(frame, f'{crack_area_percentage:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display crack count on the frame
    cv2.putText(frame, f'Wall Cracks: {crack_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    try:
        # Decode the image
        np_img = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Perform crack detection on the frame
        frame = perform_crack_detection(frame)
        ret, buffer = cv2.imencode('.jpg', frame)

        # Emit the processed frame back to the client
        emit('result', buffer.tobytes())
    except Exception as e:
        emit('error', {'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
