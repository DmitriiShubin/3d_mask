from typing import Dict

import numpy as np
import pyvista as pv


class MaskFactory:
    def __init__(self, config: Dict):

        # eval several components of dict
        config['texture_path'] = eval(config['texture_path'])
        config['mesh_color'] = eval(config['mesh_color'])
        config['background_color'] = eval(config['background_color'])

        self.axes = pv.Axes(
            show_actor=True, actor_scale=2.0, line_width=5
        )  # center of coordinates, will need for rotation

        # load the mesh
        self.mesh = pv.read(config['model_path'])
        if config['texture_path'] is None:
            self.texture = None  # optional
            self.mesh_color = config['mesh_color']
        else:
            self.texture = pv.read_texture(config['texture_path'])
            self.mesh_color = None

        # center and scale the mesh
        points = self.mesh.extract_feature_edges()
        points = points.points
        self.mesh.translate(-1 * np.mean(points, axis=0))

        points = self.mesh.extract_feature_edges()
        points = points.points
        self.mesh.scale(1 / np.max(np.abs(points)))

        # set up reference eye points
        self.eye_points = pv.PolyData(np.array(config['eye_coordinates']))

        # rotate mesh to have a front look
        self.mesh.rotate_x(config['initial_rotation']['x'], point=self.axes.origin)
        self.eye_points.rotate_x(config['initial_rotation']['x'], point=self.axes.origin)

        self.mesh.rotate_y(config['initial_rotation']['y'], point=self.axes.origin)
        self.eye_points.rotate_y(config['initial_rotation']['y'], point=self.axes.origin)

        self.mesh.rotate_z(config['initial_rotation']['z'], point=self.axes.origin)
        self.eye_points.rotate_z(config['initial_rotation']['z'], point=self.axes.origin)

        # define baclground color
        self.background_color = config['background_color']

        # define threshold
        self.threshold = config['color_threshold']

    def get_mesh_texture(self):
        return self.texture

    def get_mesh_color(self):
        return self.mesh_color

    def get_threshold(self):
        return self.threshold

    def get_background_color(self):
        return self.background_color

    def get_rotated_mesh_and_points(
        self, theta_x: float, theta_y: float, theta_z: float
    ) -> [pv.DataSet, pv.PolyData]:

        mesh_r = self.mesh.copy()

        mesh_r.rotate_x(theta_x, point=self.axes.origin)
        mesh_r.rotate_y(theta_y, point=self.axes.origin)
        mesh_r.rotate_z(theta_z, point=self.axes.origin)

        eye_points_r = self.eye_points.copy()

        eye_points_r.rotate_x(theta_x, point=self.axes.origin)
        eye_points_r.rotate_y(theta_y, point=self.axes.origin)
        eye_points_r.rotate_z(theta_z, point=self.axes.origin)

        return mesh_r, eye_points_r
