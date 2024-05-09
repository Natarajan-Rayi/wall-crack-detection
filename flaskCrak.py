from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import numpy as np
import base64
import shutil
import os
from flask_socketio import SocketIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "FLASK!"
socketio = SocketIO(app)

# Global variables to store video capture, processing status, and whether video streaming is active
video_capture = None
processing_status = "Not Started"
video_streaming_active = False
crack_count = 0


# Function to perform crack detection
def detect_cracks(video_path):
    # Read the video
    video_capture = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = 'output_video_with_crack_detection.avi'
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(video_capture.get(3)), int(video_capture.get(4))))

    # Initialize crack count
    crack_count = 0

    # Loop through each frame
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initial crack detection
        edges = cv2.Canny(gray, 50, 150)

        # Calculate percentage of crack
        crack_percentage = calculate_crack_percentage(edges)

        # Find contours of cracks
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)

            # Filter out small contours (noise)
            if area < 100:
                continue

            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out contours not in the top part of the image (assuming wall cracks)
            if y + h < frame.shape[0] // 2:
                continue

            # Draw bounding box around crack
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Increment crack count
            crack_count += 1

            # Display crack count
            cv2.putText(frame, f'Crack Count: {crack_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display crack percentage
            cv2.putText(frame, f'Crack Percentage: {crack_percentage:.2f}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame into the output video
        out.write(frame)

    # Release video capture and writer
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path

# Function to calculate percentage of crack in the frame
def calculate_crack_percentage(edges):
    total_pixels = edges.shape[0] * edges.shape[1]
    non_zero_pixels = np.count_nonzero(edges)
    percentage = (non_zero_pixels / total_pixels) * 100
    return percentage

def perform_crack_detection_vid(frame):
    global crack_count

    # Reset crack count for each frame
    crack_count = 0

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to separate cracks from background
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                    total_area = frame.shape[0] * frame.shape[1]
                    crack_area_percentage = (area / total_area) * 100

                    # Display percentage on the frame
                    cv2.putText(frame, f'{crack_area_percentage:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display crack count on the frame
    cv2.putText(frame, f'Wall Cracks: {crack_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


def perform_crack_detection(frame):
    global crack_count

    # Reset crack count for each frame
    crack_count = 0

    # Calculate total area of the frame
    total_area = frame.shape[0] * frame.shape[1]

    # Your crack detection code goes here
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


def gen_frames():
    global video_capture
    while True:
        if video_capture is None:
            break
        success, frame = video_capture.read()  # Read a frame from the camera
        if not success:
            break
        else:
            # Perform crack detection on the frame
            frame = perform_crack_detection(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('initial.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/live-stream')
def livestrem():
    return render_template('index.html')

@app.route('/video-detection')
def videodetection():
    return render_template('videoUpload.html')

@app.route('/image-detection')
def imagedetection():
    return render_template('imageUpload.html')

@app.route('/start_video', methods=['POST'])
def start_video():
    global video_capture, processing_status, video_streaming_active

    if video_streaming_active:
        return jsonify({"error": "Video stream is already active"})

    video_capture = cv2.VideoCapture(0)  # Open the default camera
    processing_status = "Processing..."
    video_streaming_active = True

    return jsonify({"message": "Video stream started"})

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global video_capture, processing_status, video_streaming_active

    if not video_streaming_active:
        return jsonify({"error": "Video stream is not active"})

    video_streaming_active = False
    processing_status = "Not Started"

    if video_capture:
        video_capture.release()
        video_capture = None  # Reset the video capture object

    return jsonify({"message": "Video stream stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

import base64

@app.route('/upload_and_detect', methods=['POST'])
def upload_and_detect():
    global processing_status, crack_count

    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Check if the file is allowed based on its extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file format"})

    # Read the file content
    file_content = file.read()

    if file.filename.split('.')[-1].lower() in {'png', 'jpg', 'jpeg'}:
        # Process image
        nparr = np.frombuffer(file_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_img = perform_crack_detection(img)

        # Encode the processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"crack_count": crack_count, "processed_image_base64": f"data:image/jpeg;base64,{img_str}"})

    elif file.filename.split('.')[-1].lower() in {'mp4', 'avi', 'mov'}:
            import os
            # Process video
            # Save the video file temporarily
            with open('uploaded_video.' + file.filename.split('.')[-1], 'wb') as f:
                f.write(file.read())

            video_capture = cv2.VideoCapture('uploaded_video.' + file.filename.split('.')[-1])

            # Process the video frames
            frames_with_cracks = []
            while True:
                success, frame = video_capture.read()
                if not success:
                    break
                processed_frame = perform_crack_detection(frame)
                frames_with_cracks.append(processed_frame)

            # Encode processed frames to base64
            encoded_frames = []
            for frame in frames_with_cracks:
                _, buffer = cv2.imencode('.jpg', frame)
                img_str = base64.b64encode(buffer).decode('utf-8')
                encoded_frames.append(img_str)

            # Release video capture object and delete temporary file
            video_capture.release()
            os.remove('uploaded_video.' + file.filename.split('.')[-1])

            return jsonify({"crack_count": crack_count, "processed_frames_base64": encoded_frames})

    else:
        return jsonify({"error": "Unsupported file format"})

@app.route('/upload_and_detect_video', methods=['POST'])
def upload_and_detect_video():
    import os
    global processing_status, crack_count

    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Check if the file is allowed based on its extension
    allowed_extensions = {'mp4', 'avi', 'mov'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file format"})

    try:
        # Save the video file temporarily
        video_filename = 'uploaded_video.' + file.filename.split('.')[-1]
        file.save(video_filename)

        video_capture = cv2.VideoCapture(video_filename)

        # Process the video frames
        frames_with_cracks = []
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            processed_frame = perform_crack_detection_vid(frame)
            frames_with_cracks.append(processed_frame)

        # Encode processed frames to base64
        encoded_frames = []
        for frame in frames_with_cracks:
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            encoded_frames.append(img_str)

        # Release video capture object and delete temporary file
        video_capture.release()
        os.remove(video_filename)

        return jsonify({"crack_count": crack_count, "processed_frames_base64": encoded_frames})

    except Exception as e:
        return jsonify({"error": str(e)})


# Route to handle video upload and processing
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No file part'
    
    video_file = request.files['video']
    if video_file.filename == '':
        return 'No selected file'

    # Save the uploaded video
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    video_path = os.path.join(uploads_dir, video_file.filename)
    video_file.save(video_path)

    # Perform crack detection
    processed_video_path = detect_cracks(video_path)

    # Remove uploaded video
    os.remove(video_path)
    shutil.rmtree(uploads_dir)

    # Send the processed video back to the client
    return send_file(processed_video_path, as_attachment=True)



if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
