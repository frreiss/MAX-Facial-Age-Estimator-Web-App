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


def box_color(frames_since_update):
    """
    Compute the color of the bounding box.

    The color fades from red to yellow as we get further from an inference
    result.

    Args:
        frames_since_update: How many frames have been displayed since
            updating the age
    """
    HOT_COLOR = np.array([255., 0., 0.])
    COLD_COLOR = np.array([255., 255., 0.])
    DECAY_TIME_FRAMES = 10

    if frames_since_update > DECAY_TIME_FRAMES:
        return tuple(COLD_COLOR)
    else:
        cold_weight = frames_since_update / DECAY_TIME_FRAMES
        hot_weight = 1.0 - cold_weight
        color = hot_weight * HOT_COLOR + cold_weight * COLD_COLOR
        return tuple(color)


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y),
                  (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)


def draw_boxes_and_label(image, label, box, color=(255, 255, 0)):
    """
    Modify an image by inserting a labeled bounding box.

    Args:
        image: The original image as a numpy ndarray
        label: Text label string to apply to the box
        box: list of integers (x1, y1, x2, y2) that describe the
             coordinates of the upper left corner and the width
             and height of the box
        color: Tuple of RGB values to use as the color of the box
    Returns the original image, with the indicated box drawn
    """
    x1, y1, x2, y2 = (int(c) for c in box)
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
    TARGET_FPS = 20.0
    FRAME_TIME_INTERVAL = 1.0 / TARGET_FPS
    IMAGE_WIDTH_PX = 1024  # Width of image AFTER resizing

    # Factor to use for exponentially decaying averages.
    # 0.0 => ignore new values, 1.0 => ignore old values
    EXP_DECAY_FACTOR = 0.1

    frames_per_second = 0           # Stores FPS of last frame processed
    frames_since_update = 0

    tracker = cv2.MultiTracker_create()
    executor = ThreadPoolExecutor(max_workers=1)
    future = None

    # The image submitted for the most recent inference request
    last_inference_image = None

    # List of images captured since the last time an image was submitted to
    # the expensive backend model.
    images_since_submit = []

    # Bounding boxes of faces in the most recent frame
    bounding_boxes = []

    # Age estimates corresponding to the bounding boxes.
    # When more than one age has been received, these are exponentially
    # decaying averages
    age_results = []

    while True:
        start = time.time()
        try:
            input_img = base64_to_pil_image(app.queue.get_nowait().split('base64')[-1])
        except queue.Empty:
            input_img = base64_to_pil_image(app.queue.get().split('base64')[-1])

        img_np_frame = np.array(input_img)
        img_h, img_w, _ = np.shape(img_np_frame)
        img_np_frame = cv2.resize(img_np_frame,
                                  (IMAGE_WIDTH_PX, int(IMAGE_WIDTH_PX*img_h/img_w)))
        img_h, img_w, _ = np.shape(img_np_frame)

        # Start by handling any outstanding results from previous model
        # invocations.
        if future is not None and future.done():
            # Remember the previous results so we can connect them with the
            # new results.
            last_inference_image_bounding_boxes = bounding_boxes
            last_inference_image_ages = age_results

            predict_results = future.result()
            bounding_boxes = [entry['face_box'] for entry in predict_results]
            age_results = [entry['age_estimation'] for entry in predict_results]
            tracker = update_trackers(last_inference_image, bounding_boxes)

            # Play back the video that has happened since the image was
            # submitted for inference, updating the bounding boxes as we go
            for img in images_since_submit:
                _, _ = tracker.update(img)

            # Match faces from the previous match with the current match
            bbox_mapping = match_bounding_boxes(last_inference_image_bounding_boxes,
                                                bounding_boxes)

            for old_ix, new_ix in bbox_mapping:
                old_age = last_inference_image_ages[old_ix]
                new_age = age_results[new_ix]
                exp_decay_average_age = (new_age * EXP_DECAY_FACTOR
                                         + old_age * (1.0 - EXP_DECAY_FACTOR))
                age_results[new_ix] = exp_decay_average_age

            frames_since_update = 0
        else:
            frames_since_update += 1

        # Start a new inference request if it is appropriate to do so.
        if future is None or future.done():
            future = executor.submit(predict_age_local, img_np_frame)
            last_inference_image = img_np_frame
            images_since_submit.clear()
        else:
            images_since_submit.append(img_np_frame)

        # Use CV2 MultiTracker to track faces and pair ages to face
        # For now, every box gets the same color.
        color_tuple = box_color(frames_since_update)
        success, bounding_boxes = tracker.update(img_np_frame)
        for i, (box, age) in enumerate(zip(bounding_boxes, age_results)):
            img_np_frame = draw_boxes_and_label(img_np_frame, str(int(age)), box,
                                                color_tuple)

        # draw_FPS(img_np_frame, frames_per_second)
        result_image = convert_to_JPEG(img_np_frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + result_image + b'\r\n')

        loop_process_time = time.time() - start
        if loop_process_time < FRAME_TIME_INTERVAL:
            time.sleep(FRAME_TIME_INTERVAL - loop_process_time)

        actual_loop_time = time.time() - start
        frames_per_second = 1.0 / actual_loop_time
        print("Processing time {:4.3f} sec; FPS {:4.2f}"
              "".format(loop_process_time, frames_per_second))


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
        # Old code was:
        # tracker.add(cv2.TrackerKCF_create(), image, tuple(box))
        # We use MedianFlow tracker now because it is faster. Even though the
        # algorithm is less accurate, results are more accurate because we drop
        # fewer frames.
        tracker.add(cv2.TrackerMedianFlow_create(), image, tuple(box))
    return tracker


def match_bounding_boxes(old_bounding_boxes, new_bounding_boxes):
    """
    Find matches between two sets of bounding boxes.

    We currently match based on distance between the centers of the boxes.

    Args:
        old_bounding_boxes: List of lists of bounding box coords, where
            each entry is in the format [x1, y1, x2, y2] and describes the
            upper-left corner and width and height of the box
        new_bounding_boxes: Second list of bounding boxes in the same format
            as the first

    Returns a list of tuples: (old_ix, new_ix), where old_ix and new_ix are
    offsets into the old and new bounding boxes and each pair indicates a
    match between the bounding boxes at the indicated offsets.
    """
    DISTANCE_THRESH = 200
    BIG_DISTANCE = 1e6
    BIG_DISTANCE_MINUS_EPSILON = 9e5

    def center(box):
        x1, y1, x2, y2 = tuple(box)
        return [x1 + (x2 / 2), y1 + (y2 / 2)]

    # O(n^2) comparison operation, so use numpy to scale n as far as we can
    old_centers = np.array([center(b) for b in old_bounding_boxes])
    new_centers = np.array([center(b) for b in new_bounding_boxes])

    if old_centers.shape[0] == 0 or new_centers.shape[0] == 0:
        # Special-case: One of input lists is empty
        return []

    # Generate table of euclidean distances, indexed by (old, new)
    all_diffs = np.sqrt(np.sum(np.square(old_centers[:, None] - new_centers), axis=2))

    # Replace everything over the threshold with a large number
    all_diffs[all_diffs > DISTANCE_THRESH] = BIG_DISTANCE

    # Pick the best match for each NEW bounding box.
    matches = np.argmin(all_diffs, axis=0)

    # Filter out the matches that didn't satisfy our threshold constraint
    min_distances = np.min(all_diffs, axis=0)
    matches[np.argwhere(min_distances > BIG_DISTANCE_MINUS_EPSILON)] = -1

    results_as_dict = {}
    for new_ix in range(matches.shape[0]):
        old_ix = matches[new_ix]
        if old_ix == -1:
            pass
        elif old_ix not in results_as_dict:
            results_as_dict[old_ix] = new_ix
        else:
            # Break ties by distance
            my_distance = all_diffs[old_ix, new_ix]
            their_distance = all_diffs[old_ix, results_as_dict[old_ix]]
            if my_distance < their_distance:
                results_as_dict[old_ix] = new_ix

    ret = [(k, v) for k, v in results_as_dict.items()]
    return ret


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=7000)
