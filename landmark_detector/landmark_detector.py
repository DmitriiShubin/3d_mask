import mediapipe as mp
import cv2

class LandmarkDetector:

    def __init__(self,config):


        self.faceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=config['num_faces'])

    def run(self,image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions = self.faceMesh.process(image)


        image_x_shape = image.shape[1]
        image_y_shape = image.shape[0]

        if predictions.multi_face_landmarks is not None:
            landmarks = {}
            landmarks['left_eye'] = (int(predictions.multi_face_landmarks[0].landmark[33].x*image_x_shape),
                                     int(predictions.multi_face_landmarks[0].landmark[33].y*image_y_shape)
                                     )
            landmarks['right_eye'] = (int(predictions.multi_face_landmarks[0].landmark[263].x *image_x_shape),
                                     int(predictions.multi_face_landmarks[0].landmark[263].y * image_y_shape)
                                     )
            landmarks['nose'] = (int(predictions.multi_face_landmarks[0].landmark[0].x * image_x_shape),
                                      int(predictions.multi_face_landmarks[0].landmark[0].y * image_y_shape)
                                      )
            landmarks['forehead'] = (int(predictions.multi_face_landmarks[0].landmark[151].x * image_x_shape),
                                 int(predictions.multi_face_landmarks[0].landmark[151].y * image_y_shape)
                                 )


            return landmarks
        else:
            return None