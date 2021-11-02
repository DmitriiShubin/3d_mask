from typing import List, Tuple

import numpy as np
import pyvista as pv


class Renderer:
    def __init__(self, img_width: int, img_height: int, background_color: Tuple[float, float, float]):

        self.img_width = img_width
        self.img_height = img_height

        # Pick a background colour (default is light gray)
        self.background_color = background_color  # RGB

        # setup the camera
        self.camera = pv.Camera()
        self.camera.position = [0, 0, 10]
        near_range = 0.1
        far_range = 10
        self.camera.clipping_range = (near_range, far_range)
        self.camera.view_angle = 16

        # set up precomputed camera and projection matrixes
        self.modelTransform = np.array(
            [[1.0, 0.0, 0.0, -0.0], [0.0, 1.0, 0.0, -0.0], [0.0, 0.0, 1.0, -10.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        self.projTransform = np.array(
            [
                [5.3365273, 0.0, 0.0, 0.0],
                [0.0, 7.11537, 0.0, 0.0],
                [0.0, 0.0, -1.0, -0.2],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )

    def calculate_reference_point_projections(self, eye_points: pv.DataSet) -> List:

        projections = []

        points = np.array(eye_points.points)
        for i in range(points.shape[0]):
            point = points[i]
            point = np.concatenate([point, np.ones(1)], axis=0)

            projection = np.matmul(np.matmul(self.projTransform, self.modelTransform), point)
            projection /= projection[-2]  # divide by Z
            projection[1] /= -1  # Y is negative
            projection[0] *= self.img_width // 2  # scale up
            projection[1] *= self.img_height // 2  # scale up

            projections.append(
                (int(self.img_width // 2 + projection[0]), int(self.img_height // 2 + projection[1]))
            )

        return projections

    def render_mesh_to_image(self, mesh: pv.DataSet):

        # create renderer
        pl = pv.Plotter(off_screen=True)
        pl.store_image = True
        pl.window_size = self.img_width, self.img_height
        pl.background_color = self.background_color
        pl.camera = self.camera

        # add object
        pl.add_mesh(mesh, color=[0.2, 0.2, 0.2])

        # render the image
        pl.show()

        return pl.image.astype(np.float32)
