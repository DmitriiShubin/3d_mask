import open3d as o3d
import numpy as np
import copy
import cv2
from .moving_average import MovingAverage
import pickle

class Mask:
    def __init__(self, config, frame_size):

        #load models for approximation of the head position
        self.theta_y_model = pickle.load(open('./data/face_pose_models/linear_model_theta_y.pkl', 'rb'))
        self.theta_x_model = pickle.load(open('./data/face_pose_models/linear_model_theta_x.pkl', 'rb'))

        # load the mesh
        self.mesh = o3d.io.read_triangle_mesh("./data/3d_models/Mask.fbx")
        self.mesh.compute_vertex_normals()

        # center and scale the mesh
        self.mesh = self.mesh.translate(-1 * self.mesh.get_center())
        self.mesh = self.mesh.scale(
            1 / np.max(np.abs(np.asarray(self.mesh.vertices))), center=self.mesh.get_center()
        )
        # colorize
        self.mesh.paint_uniform_color(config['mesh_color'])

        # set up reference eye points
        self.eye_points = o3d.geometry.PointCloud()
        self.eye_points.points = o3d.utility.Vector3dVector(
            np.array([

                [-0.01, 0.125, 0.5],
                [-0.01, 0.125, -0.5]

            ])
        )

        # define moving averages for various components
        self.moving_average_scale = MovingAverage(length=config['ma_scale_length'])
        self.moving_average_position_x = MovingAverage(length=config['ma_position_length'])
        self.moving_average_position_y = MovingAverage(length=config['ma_position_length'])
        self.moving_average_theta_x = MovingAverage(length=config['ma_rotation_length'])
        self.moving_average_theta_y = MovingAverage(length=config['ma_rotation_length'])
        self.moving_average_theta_z = MovingAverage(length=config['ma_rotation_length'])


        # rotate points to have a front loor
        R = self.mesh.get_rotation_matrix_from_xyz((0,np.pi/2,0))
        self.mesh.rotate(R, center=(0, 0, 0))
        self.eye_points.rotate(R, center=(0, 0, 0))

        ############# parameters of scene for rendering #############

        # Define the material
        self.mtl = o3d.visualization.rendering.Material()
        self.mtl.base_color = config['render_color']
        self.mtl.shader = "defaultLit"

        self.img_width = frame_size[0]
        self.img_height = frame_size[1]
        self.render = o3d.visualization.rendering.OffscreenRenderer(self.img_width, self.img_height)

        # Pick a background colour (default is light gray)
        self.render.scene.set_background([255, 255, 255, 0.5])  # RGBA

        # Optionally set the camera field of view (to zoom in a bit)
        vertical_field_of_view = 16.0  # between 5 and 90 degrees
        aspect_ratio = self.img_width / self.img_height  # azimuth over elevation
        near_plane = 0.1
        far_plane = 10
        fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
        self.render.scene.camera.set_projection(
            vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type
        )

        # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
        center = [0, 0, 0]  # look_at target
        eye = [0, 0, 10]  # camera position
        up = [0, 1, 0]  # camera orientation
        self.render.scene.camera.look_at(center, eye, up)

    def run(self, frame, left_eye_position, right_eye_position,forehead_position,nose_position,center_position):

        mesh_r, eye_points_r = self._rotate_mesh(left_eye=left_eye_position, right_eye=right_eye_position,forehead=forehead_position,
                                                 nose=nose_position,center=center_position)

        # add object
        self.render.scene.add_geometry("rotated_model", mesh_r, self.mtl)

        # render the image
        image = self.render.render_to_image()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)

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
        # remove all objects from render
        self.render.scene.clear_geometry()

        # masking the rendered image
        mask = self._get_binary_mask(image=image)
        # alignment of images
        frame[mask == 1] = image[mask == 1]

        return frame

    ####### Utils #########

    def _calculate_reference_point_projections(self, eye_points):
        P = self.render.scene.camera.get_projection_matrix()
        V = self.render.scene.camera.get_view_matrix()

        projections = []

        points = np.array(eye_points.points)
        for i in range(points.shape[0]):
            point = points[i]
            point = np.concatenate([point, np.ones(1)], axis=0)

            projection = np.matmul(np.matmul(P, V), point)
            projection /= projection[-2]  # divide by Z
            projection[1] /= -1  # Y is negative
            projection[0] *= self.img_width // 2  # scale up
            projection[1] *= self.img_height // 2  # scale up

            projections.append(
                (int(self.img_width // 2 + projection[0]), int(self.img_height // 2 + projection[1]))
            )

        return projections

    def _get_binary_mask(self, image):
        mask = np.zeros(image.shape)
        mask[image[:, :, 0] < 150] = 1  # thresholding

        return mask

    def _scale_mesh(self, image, left_eye, right_eye, render_eye_points):

        frame_distance = self._compute_distance_between_points(x=left_eye, y=right_eye)
        render_reference_points = self._calculate_reference_point_projections(render_eye_points)
        render_distance = self._compute_distance_between_points(
            x=render_reference_points[0], y=render_reference_points[1]
        )

        scale_factor = frame_distance / render_distance
        scale_factor *= 1.4

        #apply moving average
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

    def _compute_distance_between_points(self, x, y):
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1])**2)

    def _locate_mesh(self, image, frame_reference_points, render_reference_points):

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

    def _rotate_mesh(self, left_eye, right_eye,forehead,nose,center):

        face_features = self.exract_face_features(left_eye=left_eye,right_eye=right_eye,
                                                  forehead=forehead,
                                                  nose=nose,
                                                  center=center)


        theta_y = self.theta_y_model.predict(np.array([face_features['upper_sides_proportion']]).reshape(-1,1))[0]
        theta_y = np.clip(theta_y,a_min=-30,a_max=30)
        features_x = np.array([
                               face_features['vertical_sides_proportion']

                               ])
        theta_x = self.theta_x_model.predict(features_x.reshape(-1,features_x.shape[0]))[0]

        theta_x = np.clip(theta_x,a_min=-15,a_max=15)


        theta_z = (-self.calculate_angle_x(left_eye=left_eye, right_eye=right_eye) / 180) * np.pi
        theta_y = (-theta_y / 180) * np.pi
        theta_x = (theta_x / 180) * np.pi

        theta_x = self.moving_average_theta_x.run(value=theta_x)
        theta_y = self.moving_average_theta_y.run(value=theta_y)
        theta_z = self.moving_average_theta_z.run(value=theta_z)

        R = self.mesh.get_rotation_matrix_from_xyz((theta_x, theta_y, theta_z))
        mesh_r = copy.deepcopy(self.mesh)
        mesh_r.rotate(R, center=(0, 0, 0))

        eye_points_r = copy.deepcopy(self.eye_points)
        eye_points_r.rotate(R, center=(0, 0, 0))

        return mesh_r, eye_points_r

    def exract_face_features(self,left_eye, right_eye,forehead,nose,center):

        """
        features:

        length(left_eye,fore)/length(right_eye,fore)

        angles between  eyes + forehead


        """

        features = {}

        a = self._compute_distance_between_points(left_eye,
                                             forehead
                                             )
        b = self._compute_distance_between_points(right_eye,
                                             forehead
                                             )
        c = self._compute_distance_between_points(right_eye,
                                             left_eye
                                             )

        d = self._compute_distance_between_points(forehead,
                                             center
                                             )

        e = self._compute_distance_between_points(center,
                                             nose
                                             )

        features['upper_sides_proportion'] = a / b
        features['vertical_sides_proportion'] = d / c
        features['alpha'] = np.degrees(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
        features['beta'] = np.degrees(np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))
        features['gamma'] = np.degrees(np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))

     
        return features

    def calculate_angle_x(self, left_eye: tuple, right_eye: tuple):

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
