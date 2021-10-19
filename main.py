from time import time

import click
import cv2
import yaml

from face_detector import FaceDetector
from landmark_detector import LandmarkDetector

# utils
from utils import expand_face_box


@click.command()
@click.option('--config_path', default='./config.yml', help='')
def main(config_path):

    # read configs
    config = yaml.load(open(config_path))

    cap = cv2.VideoCapture(0)

    # intialize classes
    landmark_detector = LandmarkDetector(config=config)
    face_detector = FaceDetector(model_path=config['face_detector_path'])

    while True:

        start = time()

        # get frame from video
        ret, frame = cap.read()

        # get face detection bbxes
        bboxes = face_detector.run(frame=frame)

        if bboxes.shape[0] > 0:
            bboxes = expand_face_box(
                image_x_shape=frame.shape[1],
                image_y_shape=frame.shape[0],
                bboxes=bboxes,
                face_area_coef=config['face_area_coeficient'],
            )

        for i in range(bboxes.shape[0]):
            cv2.rectangle(
                frame,
                (bboxes[i, 0], bboxes[i, 1]),
                (bboxes[i, 2], bboxes[i, 3]),
                color=(0, 0, 255),
                thickness=2,
            )

        # get face landmarks
        for i in range(bboxes.shape[0]):
            landmarks = landmark_detector.run(frame=frame, bbox=bboxes[i, :])
            if landmarks is not None:
                frame = cv2.circle(frame, landmarks['right_eye'], radius=2, color=(0, 0, 255), thickness=2)
                frame = cv2.circle(frame, landmarks['left_eye'], radius=2, color=(0, 0, 255), thickness=2)
                frame = cv2.circle(frame, landmarks['nose'], radius=2, color=(0, 0, 255), thickness=2)
                frame = cv2.circle(frame, landmarks['forehead'], radius=2, color=(0, 0, 255), thickness=2)

        end = time()

        cv2.putText(
            frame, f'FPS:{int(1/(end-start))}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return True


if __name__ == '__main__':
    main()
