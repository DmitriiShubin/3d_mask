from time import time

import click
import cv2
import yaml

from utils import MaskAlignment
from utils.landmark_detector import LandmarkDetector
import numpy as np

@click.command()
@click.option('--config_path', default='./config.yml', help='')
def main(config_path):

    # read configs
    config = yaml.load(open(config_path))

    cap = cv2.VideoCapture(0)

    # intialize classes
    landmark_detector = LandmarkDetector(config=config)

    _, frame = cap.read()
    mask = MaskAlignment(config=config, frame_size=frame.shape[:2][::-1])



    while True:

        start = time()

        # get frame from video
        ret, frame = cap.read()

        #resize to 480x640
        if not np.isclose((frame.shape[0]/frame.shape[1]),0.75,atol=0.01):
            frame = frame[:,:int(frame.shape[1]*0.75),:]

        frame = cv2.resize(frame,(640,480))



        # get face landmarks
        landmarks = landmark_detector.run(frame=frame)
        if landmarks is not None:
            if config['show_landmarks']:
                frame = cv2.circle(frame, landmarks['right_eye'], radius=2, color=(0, 0, 255), thickness=2)
                frame = cv2.circle(frame, landmarks['left_eye'], radius=2, color=(0, 0, 255), thickness=2)
                frame = cv2.circle(frame, landmarks['nose'], radius=2, color=(0, 0, 255), thickness=2)
                frame = cv2.circle(frame, landmarks['forehead'], radius=2, color=(0, 0, 255), thickness=2)
                frame = cv2.circle(frame, landmarks['center'], radius=2, color=(0, 0, 255), thickness=2)

            frame = mask.run(
                frame=frame,
                left_eye_position=landmarks['left_eye'],
                right_eye_position=landmarks['right_eye'],
                forehead_position=landmarks['forehead'],
                nose_position=landmarks['nose'],
                center_position=landmarks['center'],
                show_mask=config['show_mask'],
            )

        end = time()

        cv2.putText(
            frame, f'FPS:{int(1/(end-start))}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"Mask: {config['mask_type']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return True


if __name__ == '__main__':
    main()
