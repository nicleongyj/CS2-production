from norfair import Detection, Tracker, Video, draw_tracked_objects
import cv2
import tensorflow as tf
import numpy as np


class MyDetector:
    def __init__(self, model_path, threshold=0.5):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.threshold = threshold

    def __call__(self, frame):
        input_shape = self.input_details[0]['shape']
        input_data = cv2.resize(frame, (input_shape[1], input_shape[2]))
        input_data = np.expand_dims(input_data, axis=0)
        input_data = (input_data.astype('float32') / 255.0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        detections = self._post_process(output_data)
        print(detections)
        print(detections.shape)
        return detections

    def _post_process(self, output_data):
        # Perform post-processing here based on your model's output format
        # Convert raw output to detections (e.g., bounding boxes, classes)
        # Apply thresholding and non-maximum suppression as needed
        return output_data


detector = MyDetector(model_path='./saved_model/yolov8_hdb_float32.tflite')
video = Video(input_path="./assets/sample.mp4")
# tracker = Tracker(distance_function="euclidean", distance_threshold=100)

# for frame in video:
#    detections = detector(frame)
#    norfair_detections = [Detection(points) for points in detections]
#    tracked_objects = tracker.update(detections=norfair_detections)
#    draw_tracked_objects(frame, tracked_objects)
#    video.write(frame)

for frame in video:
    detections = detector(frame)
    break