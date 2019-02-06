#
# Copyright 2018 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import time
from concurrent.futures import ThreadPoolExecutor

import queue
import base64
import numpy as np
import requests
import json
import cv2
from flask import Flask, render_template, Response
from gevent import monkey
from io import BytesIO
from PIL import Image
try:
    from flask.ext.socketio import SocketIO, emit
except ImportError:
    from flask_socketio import SocketIO, emit


monkey.patch_all()
app = Flask(__name__)
app.config.from_object('config')
app.queue = queue.Queue(1)  # Keep queue size low to avoid video frame lag
socketio = SocketIO(app)


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y),
                  (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)


def draw_boxes_and_label(image, label, box, color=(255, 255, 0)):
    box = [int(n) for n in box]
    x1, y1, x2, y2 = tuple(box)
    p1 = (x1, y1)
    p2 = (x1 + x2, y1 + y2)
    cv2.rectangle(image, p1, p2, color, 2, 1)
    draw_label(image, p1, label)
    return image


def draw_FPS(image, fps):
    cv2.putText(image, "FPS: {}".format("%.4f" % fps), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def base64_to_pil_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))


def convert_to_JPEG(np_image_frame):
    # np_image_color = cv2.cvtColor(np_image_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(np_image_frame)
    with BytesIO() as f:
        image.save(f, format='JPEG', quality=95)
        return f.getvalue()


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template('index.html')


@socketio.on('netin', namespace='/streaming')
def msg(dta):
    emit('response', {'data': dta['data']})


@socketio.on('connected', namespace='/streaming')
def connected():
    emit('response', {'data': 'OK'})


@socketio.on('streamingvideo', namespace='/streaming')
def webdata(dta):
    try:
        app.queue.put_nowait(dta['data'])
    except queue.Full:
        pass


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen():
    TARGET_FPS = 10.0
    FRAME_TIME_INTERVAL = 1.0 / TARGET_FPS
    IMAGE_RESOLUTION = 1024

    frames_per_second = 0           # Stores FPS of last frame processed
    age_results = []

    tracker = cv2.MultiTracker_create()
    executor = ThreadPoolExecutor(max_workers=1)
    future = None

    # The image submitted for the most recent inference request
    last_inference_image = None
    
    # List of images captured since the last time an image was submitted to
    # the expensive backend model.
    images_since_submit = []

    while True:
        start = time.time()
        try:
            input_img = base64_to_pil_image(app.queue.get_nowait().split('base64')[-1])
        except queue.Empty:
            input_img = base64_to_pil_image(app.queue.get().split('base64')[-1])

        img_np_frame = np.array(input_img)
        img_h, img_w, _ = np.shape(img_np_frame)
        img_np_frame = cv2.resize(img_np_frame,
                                  (IMAGE_RESOLUTION, int(IMAGE_RESOLUTION*img_h/img_w)))
        img_h, img_w, _ = np.shape(img_np_frame)

        # Start by handling any outstanding results from previous model
        # invocations.
        if future is not None and future.done():
            predict_results = future.result()
            bounding_boxes = [entry['face_box'] for entry in predict_results]
            age_results = [entry['age_estimation'] for entry in predict_results]
            tracker = update_trackers(last_inference_image, bounding_boxes)

            # Play back the video that has happened since the image was
            # submitted for inference, updating the bounding boxes as we go
            for img in images_since_submit:
                tracker.update(img)

            # Use a different color to indicate updated bounding box
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        # Start a new inference request if it is appropriate to do so.
        if future is None or future.done():
            future = executor.submit(predict_age_local, img_np_frame)
            last_inference_image = img_np_frame
            images_since_submit.clear()
        else:
            images_since_submit.append(img_np_frame)

        # Use CV2 MultiTracker to track faces and pair ages to face
        success, bounding_boxes = tracker.update(img_np_frame)
        for i, (box, age) in enumerate(zip(bounding_boxes, age_results)):
            img_np_frame = draw_boxes_and_label(img_np_frame, str(age), box, color)

        # draw_FPS(img_np_frame, frames_per_second)
        result_image = convert_to_JPEG(img_np_frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + result_image + b'\r\n')

        loop_process_time = time.time() - start
        if loop_process_time < FRAME_TIME_INTERVAL:
            time.sleep(FRAME_TIME_INTERVAL - loop_process_time)

        actual_loop_time = time.time() - start
        frames_per_second = 1.0 / actual_loop_time
        print("FPS: " + str(frames_per_second))


def predict_age_local(np_image):
    image = convert_to_JPEG(np_image)
    my_files = {'image': image,
                'Content-Type': 'multipart/form-data',
                'accept': 'application/json'}

    r = requests.post('http://localhost:5000/model/predict',
                      files=my_files, json={"key": "value"})

    json_str = json.dumps(r.json())
    data = json.loads(json_str)
    return data['predictions']


def update_trackers(image, bounding_boxes):
    tracker = cv2.MultiTracker_create()
    for box in bounding_boxes:
        tracker.add(cv2.TrackerKCF_create(), image, tuple(box))
    return tracker


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=7000)
