import open3d as o3d
import numpy as np
import copy
import cv2


class Mask:
    def __init__(self, config, frame_size):

        # load the mesh
        self.mesh = o3d.io.read_triangle_mesh("./data/front_man.stl")
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
            np.array([[0.5, -0.01, 0.2], [-0.5, -0.01, 0.2]])
        )

        # rotate points to have a front loor
        R = self.mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
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

    def run(self, frame):

        # rotation of the mesh

        # scaling of the mesh

        # location of the mesh

        # add object
        self.render.scene.add_geometry("rotated_model", self.mesh, self.mtl)

        # render the image
        image = self.render.render_to_image()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)

        # remove all objects from render
        self.render.scene.clear_geometry()

        # masking the rendered image
        mask = self._get_binary_mask(image=image)
        frame[mask == 1] = image[mask == 1]
        # alignment of images

        return frame

    ####### Utils #########

    def _calculate_reference_point_projections(self):
        P = self.render.scene.camera.get_projection_matrix()
        V = self.render.scene.camera.get_view_matrix()

        projections = []

        points = np.array(self.eye_points.points)
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
