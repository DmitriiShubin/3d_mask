import onnxruntime as ort
import cv2
import numpy as np

class FaceDetector():

    def __init__(self,model_path):

        self.model = ort.InferenceSession(model_path)


    def run(self,image):

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = face_detector.get_inputs()[0].name
        confidences, boxes = face_detector.run(None, {input_name: image})
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)