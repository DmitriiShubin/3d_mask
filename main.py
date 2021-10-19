import cv2
from time import time

import click
import yaml

from landmark_detector import LandmarkDetector



@click.command()
@click.option('--config_path', default='./config.yml', help='')
def main(config_path):

    #read configs
    config = yaml.load(open(config_path))

    cap = cv2.VideoCapture(0)

    #intialize classes
    landmark_detector = LandmarkDetector(config=config)



    while True:

        start = time()

        #get frame from video
        ret,frame = cap.read()

        landmarks = landmark_detector.run(image=frame)

        if landmarks is not None:
            frame = cv2.circle(frame, landmarks['right_eye'], radius=2, color=(0, 0, 255), thickness=2)
            frame = cv2.circle(frame, landmarks['left_eye'], radius=2, color=(0, 0, 255), thickness=2)
            frame = cv2.circle(frame, landmarks['nose'], radius=2, color=(0, 0, 255), thickness=2)
            frame = cv2.circle(frame, landmarks['forehead'], radius=2, color=(0, 0, 255), thickness=2)




        end = time()


        cv2.putText(frame, f'FPS:{int(1/(end-start))}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return True



if __name__ == '__main__':
    main()


