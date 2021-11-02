import cv2
import mediapipe as mp
import numpy as np


class LandmarkDetector:
    def __init__(self, config: dict):

        self.faceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=config['num_faces'])

    def run(self, frame: np.array) -> dict:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = self.faceMesh.process(frame)

        image_x_shape = frame.shape[1]
        image_y_shape = frame.shape[0]

        if predictions.multi_face_landmarks is not None:
            landmarks = {}
            landmarks['left_eye'] = (
                int(predictions.multi_face_landmarks[0].landmark[33].x * image_x_shape),
                int(predictions.multi_face_landmarks[0].landmark[33].y * image_y_shape),
            )
            landmarks['right_eye'] = (
                int(predictions.multi_face_landmarks[0].landmark[263].x * image_x_shape),
                int(predictions.multi_face_landmarks[0].landmark[263].y * image_y_shape),
            )
            landmarks['nose'] = (
                int(predictions.multi_face_landmarks[0].landmark[1].x * image_x_shape),
                int(predictions.multi_face_landmarks[0].landmark[1].y * image_y_shape),
            )
            landmarks['forehead'] = (
                int(predictions.multi_face_landmarks[0].landmark[151].x * image_x_shape),
                int(predictions.multi_face_landmarks[0].landmark[151].y * image_y_shape),
            )

            landmarks['center'] = (
                (landmarks['forehead'][0] + landmarks['nose'][0]) // 2,
                (landmarks['left_eye'][1] + landmarks['right_eye'][1]) // 2,
            )

            return landmarks
        else:
            return None
