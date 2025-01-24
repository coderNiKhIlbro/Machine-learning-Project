from tensorflow import keras
import cv2 
import numpy as np
from flask import Flask, render_template, Response, jsonify

# Load the pre-trained model
model = keras.models.load_model("mask_model95.5.h5")

app = Flask(__name__)

# Global variables
video_stream_active = False
camera = cv2.VideoCapture(0)

def preprocess_image(frame):
    """ Preprocess the image to match model input. """
    frame = cv2.resize(frame, (224, 224))  # Resize to match model input
    frame = frame.astype('float32') / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return frame

# Function to generate video frames
def generate_frames():
    global video_stream_active, camera
    while True:
        success, frame = camera.read()  # Read the camera frame
        if not success:
            break
        else:
            processed_frame = preprocess_image(frame)
            prediction = model.predict(processed_frame)
            predicted_class = 'Mask On' if prediction[0] > 0.5 else 'Please Wear Mask'

            # Add text to the frame
            cv2.putText(frame, f"Prediction: {predicted_class}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield frame in HTTP format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """ Render the HTML page. """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """ Provide video stream to the frontend. """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/start_video')
# def start_video():
#     """ Start video stream by setting the flag. """
#     global video_stream_active
#     video_stream_active = True
#     return jsonify({"status": "Video stream started"}), 200

# @app.route('/stop_video')
# def stop_video():
#     """ Stop video stream and release the camera. """
#     global video_stream_active, camera
#     video_stream_active = False
#     camera.release()  # Release the camera
#     camera = cv2.VideoCapture(0)  # Re-initialize for future use
#     return jsonify({"status": "Video stream stopped"}), 200

if __name__ == '__main__':
    app.run(debug=True)
