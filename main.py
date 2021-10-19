import cv2
from time import time
import mediapipe as mp
import numpy as np

NUM_FACE = 1

def calculate_angle_x(left_eye:tuple,right_eye:tuple):

    x = np.array(left_eye)
    y = np.array(right_eye)

    proj = x[1]-y[1]

    if proj > 0:
        pos = True
    elif proj < 0:
        pos = False
    else:
        return 0

    proj = np.abs(proj)

    ct = 1/np.tan(proj/(x[0]-y[0]))

    if pos:
        return 90-np.degrees(np.arctan(ct))
    else:
        return np.degrees(np.arctan(ct))-90

def main():

    cap = cv2.VideoCapture(0)

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    while True:

        start = time()

        #get frame from video
        ret,frame = cap.read()

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks is not None:
            landmarks = {}
            landmarks['left_eye'] = (int(results.multi_face_landmarks[0].landmark[33].x*frame.shape[1]),
                                     int(results.multi_face_landmarks[0].landmark[33].y*frame.shape[0])
                                     )
            landmarks['right_eye'] = (int(results.multi_face_landmarks[0].landmark[263].x * frame.shape[1]),
                                     int(results.multi_face_landmarks[0].landmark[263].y * frame.shape[0])
                                     )
            landmarks['nose'] = (int(results.multi_face_landmarks[0].landmark[0].x * frame.shape[1]),
                                      int(results.multi_face_landmarks[0].landmark[0].y * frame.shape[0])
                                      )
            landmarks['forehead'] = (int(results.multi_face_landmarks[0].landmark[151].x * frame.shape[1]),
                                 int(results.multi_face_landmarks[0].landmark[151].y * frame.shape[0])
                                 )

            frame = cv2.circle(frame, landmarks['right_eye'], radius=2, color=(0, 0, 255), thickness=2)
            frame = cv2.circle(frame, landmarks['left_eye'], radius=2, color=(0, 0, 255), thickness=2)
            frame = cv2.circle(frame, landmarks['nose'], radius=2, color=(0, 0, 255), thickness=2)
            frame = cv2.circle(frame, landmarks['forehead'], radius=2, color=(0, 0, 255), thickness=2)

            angle = calculate_angle_x(landmarks['right_eye'],landmarks['left_eye'])

            nose_eye = landmarks['nose'][1]-((landmarks['right_eye'][1]+landmarks['left_eye'][1])*0.5)
            forehead_eye = ((landmarks['right_eye'][1] + landmarks['left_eye'][1]) * 0.5)-landmarks['forehead'][1]

            #print(f"X ange: {angle}")
            print(f"Proportions: {nose_eye/forehead_eye}")


        # if results.multi_face_landmarks:
        #     for faceLms in results.multi_face_landmarks:
        #         mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)





        end = time()

        fps = 1/(end-start)

        cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return True



if __name__ == '__main__':
    main()


