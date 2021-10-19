import cv2
import mediapipe as mp


class LandmarkDetector:
    def __init__(self, config):

        self.faceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=config['num_faces'])

    def run(self, frame, bbox):

        frame = frame[bbox[0] : bbox[2], bbox[1] : bbox[3]]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = self.faceMesh.process(frame)

        image_x_shape = frame.shape[1]
        image_y_shape = frame.shape[0]

        if predictions.multi_face_landmarks is not None:
            landmarks = {}
            landmarks['left_eye'] = (
                int(predictions.multi_face_landmarks[0].landmark[33].x * image_x_shape + bbox[1]),
                int(predictions.multi_face_landmarks[0].landmark[33].y * image_y_shape + bbox[0]),
            )
            landmarks['right_eye'] = (
                int(predictions.multi_face_landmarks[0].landmark[263].x * image_x_shape + bbox[1]),
                int(predictions.multi_face_landmarks[0].landmark[263].y * image_y_shape + bbox[0]),
            )
            landmarks['nose'] = (
                int(predictions.multi_face_landmarks[0].landmark[0].x * image_x_shape + bbox[1]),
                int(predictions.multi_face_landmarks[0].landmark[0].y * image_y_shape + bbox[0]),
            )
            landmarks['forehead'] = (
                int(predictions.multi_face_landmarks[0].landmark[151].x * image_x_shape + bbox[1]),
                int(predictions.multi_face_landmarks[0].landmark[151].y * image_y_shape + bbox[0]),
            )

            return landmarks
        else:
            return None
