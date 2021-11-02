
import pickle
from typing import List, Tuple

import cv2
import numpy as np
import pyvista as pv

from .mask_factory import MaskFactory
from .moving_average import MovingAverage
from .renderer import Renderer


class MaskAlignment:
    def __init__(self, config: dict, frame_size: Tuple[int, int]):

        # load models for approximation of the head position
        self.theta_y_model = pickle.load(open('./data/face_pose_models/linear_model_theta_y.pkl', 'rb'))
        self.theta_x_model = pickle.load(open('./data/face_pose_models/linear_model_theta_x.pkl', 'rb'))

        self.mask = MaskFactory()

        # define moving averages for various components
        self.moving_average_scale = MovingAverage(length=config['ma_scale_length'])
        self.moving_average_position_x = MovingAverage(length=config['ma_position_length'])
        self.moving_average_position_y = MovingAverage(length=config['ma_position_length'])
        self.moving_average_theta_x = MovingAverage(length=config['ma_rotation_length'])
        self.moving_average_theta_y = MovingAverage(length=config['ma_rotation_length'])
        self.moving_average_theta_z = MovingAverage(length=config['ma_rotation_length'])

        self.img_width = frame_size[0]
        self.img_height = frame_size[1]

        self.renderer = Renderer(
            img_width=self.img_width, img_height=self.img_height, background_color=(1, 1, 1)
        )

    def run(
        self,
        frame: np.array,
        left_eye_position: Tuple[int, int],
        right_eye_position: Tuple[int, int],
        forehead_position: Tuple[int, int],
        nose_position: Tuple[int, int],
        center_position: Tuple[int, int],
        show_mask: bool,
    ) -> np.array:

        mesh_r, eye_points_r = self._rotate_mesh(
            left_eye=left_eye_position,
            right_eye=right_eye_position,
            forehead=forehead_position,
            nose=nose_position,
            center=center_position,
        )

        image = self.renderer.render_mesh_to_image(mesh=mesh_r)

        # apply scaling
        image, render_reference_points = self._scale_mesh(
            image=image,
            left_eye=left_eye_position,
            right_eye=right_eye_position,
            render_eye_points=eye_points_r,
        )

        # locate mesh
        image = self._locate_mesh(
            image=image,
            frame_reference_points=(left_eye_position, right_eye_position),
            render_reference_points=render_reference_points,
        )

        # masking the rendered image
        mask = self._get_binary_mask(image=image)
        if show_mask:
            # alignment of images
            frame[mask == 1] = image[mask == 1]

        return frame

    ####### Utils #########

    def _get_binary_mask(self, image: np.array) -> np.array:
        mask = np.zeros(image.shape)
        mask[image[:, :, 0] < 150] = 1  # thresholding
        return mask

    def _scale_mesh(
        self,
        image: np.array,
        left_eye: Tuple[int, int],
        right_eye: Tuple[int, int],
        render_eye_points: pv.DataSet,
    ) -> [np.array, List]:

        frame_distance = self._compute_distance_between_points(x=left_eye, y=right_eye)

        render_reference_points = self.renderer.calculate_reference_point_projections(
            eye_points=render_eye_points
        )

        render_distance = self._compute_distance_between_points(
            x=render_reference_points[0], y=render_reference_points[1]
        )

        scale_factor = frame_distance / render_distance
        scale_factor *= 1.4

        # apply moving average
        scale_factor = self.moving_average_scale.run(value=scale_factor)

        render_reference_points = [
            list(int(point * scale_factor) for point in point_pair) for point_pair in render_reference_points
        ]

        image_size = image.shape

        image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))

        if image.shape[0] <= image_size[0] and image.shape[1] <= image_size[1]:

            filled_image = np.ones(image_size) * 255
            filled_image[: image.shape[0], : image.shape[1], :] = image
        else:
            diff_x = image.shape[0] - image_size[0]
            diff_y = image.shape[1] - image_size[1]

            filled_image = image[diff_x // 2 : -1 * diff_x // 2, diff_y // 2 : -1 * diff_y // 2, :]

        return filled_image, render_reference_points

    def _compute_distance_between_points(self, x: np.array, y: np.array) -> float:
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    def _locate_mesh(
        self, image: np.array, frame_reference_points: Tuple, render_reference_points: List
    ) -> np.array:

        # compute center between reference_points
        frame_center = (
            (frame_reference_points[0][0] + frame_reference_points[1][0]) // 2,
            (frame_reference_points[0][1] + frame_reference_points[1][1]) // 2,
        )
        render_center = (
            (render_reference_points[0][0] + render_reference_points[1][0]) // 2,
            (render_reference_points[0][1] + render_reference_points[1][1]) // 2,
        )

        shift = [
            frame_center[0] - render_center[0],
            frame_center[1] - render_center[1],
        ]

        shift[0] = int(self.moving_average_position_x.run(value=shift[0]))
        shift[1] = int(self.moving_average_position_y.run(value=shift[1]))

        if shift[0] > 0:
            image[:, shift[0] :, :] = image[:, : -shift[0], :]
            image[:, : shift[0], :] = np.ones_like(image[:, : shift[0], :]) * 255
        elif shift[0] < 0:
            image[:, : shift[0], :] = image[:, -shift[0] :, :]
            image[:, shift[0] :, :] = np.ones_like(image[:, shift[0] :, :]) * 255

        if shift[1] > 0:
            image[shift[1] :, :, :] = image[: -shift[1], :, :]
            image[: shift[1], :, :] = np.ones_like(image[: shift[1], :, :]) * 255
        elif shift[1] < 0:
            image[: shift[1], :, :] = image[-shift[1] :, :, :]
            image[shift[1] :, :, :] = np.ones_like(image[shift[1] :, :, :]) * 255

        return image

    def _rotate_mesh(
        self,
        left_eye: Tuple[int, int],
        right_eye: Tuple[int, int],
        forehead: Tuple[int, int],
        nose: Tuple[int, int],
        center: Tuple[int, int],
    ) -> [pv.DataSet, pv.PolyData]:

        face_features = self.exract_face_features(
            left_eye=left_eye, right_eye=right_eye, forehead=forehead, nose=nose, center=center
        )

        theta_y = -self.theta_y_model.predict(
            np.array([face_features['upper_sides_proportion']]).reshape(-1, 1)
        )[0]
        theta_y = np.clip(theta_y, a_min=-30, a_max=30)

        features_x = np.array([face_features['vertical_sides_proportion']])
        theta_x = self.theta_x_model.predict(features_x.reshape(-1, features_x.shape[0]))[0]
        theta_x = np.clip(theta_x, a_min=-15, a_max=15)

        theta_z = -self.calculate_angle_x(left_eye=left_eye, right_eye=right_eye)

        theta_x = self.moving_average_theta_x.run(value=theta_x)
        theta_y = self.moving_average_theta_y.run(value=theta_y)
        theta_z = self.moving_average_theta_z.run(value=theta_z)

        return self.mask.get_rotated_mesh_and_points(theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)

    def exract_face_features(
        self,
        left_eye: Tuple[int, int],
        right_eye: Tuple[int, int],
        forehead: Tuple[int, int],
        nose: Tuple[int, int],
        center: Tuple[int, int],
    ) -> dict:

        """
        features:

        length(left_eye,fore)/length(right_eye,fore)

        angles between  eyes + forehead


        """

        features = {}

        a = self._compute_distance_between_points(left_eye, forehead)
        b = self._compute_distance_between_points(right_eye, forehead)
        c = self._compute_distance_between_points(right_eye, left_eye)

        d = self._compute_distance_between_points(forehead, center)

        features['upper_sides_proportion'] = a / b
        features['vertical_sides_proportion'] = d / c
        features['alpha'] = np.degrees(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
        features['beta'] = np.degrees(np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))
        features['gamma'] = np.degrees(np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))

        return features

    def calculate_angle_x(self, left_eye: Tuple[int, int], right_eye: Tuple[int, int]) -> float:

        x = np.array(right_eye)
        y = np.array(left_eye)

        proj = x[1] - y[1]

        if proj > 0:
            pos = True
        elif proj < 0:
            pos = False
        else:
            return 0

        proj = np.abs(proj)

        ct = 1 / np.tan(proj / (x[0] - y[0]))

        if pos:
            return 90 - np.degrees(np.arctan(ct))
        else:
            return np.degrees(np.arctan(ct)) - 90
