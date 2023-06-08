import math
import typing
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import KDTree


class Point:
    def __init__(self, x, y, z, r, i, a, _system_access: bool = False):
        self.x, self.y, self.z, self.r, self.incl, self.azim = x, y, z, r, i, a
        if not _system_access:
            raise PermissionError("Use 'from_cart' or 'from_pol'")

    @staticmethod
    def cart(x: float, y: float, z: float) -> "Point":
        r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        incl = math.acos(z / r)
        azim = math.atan2(y, x)
        return Point(x, y, z, r, incl, azim, _system_access=True)

    @staticmethod
    def pol(r: float, incl: float, azim: float) -> "Point":
        x = r * math.sin(math.radians(incl)) * math.cos(math.radians(azim))
        y = r * math.sin(math.radians(incl)) * math.sin(math.radians(azim))
        z = r * math.cos(math.radians(incl))
        return Point(x, y, z, r, incl, azim, _system_access=True)

    def to_cart(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    def to_pol(self) -> Tuple[float, float, float]:
        return self.r, self.incl, self.azim


class Points:
    @staticmethod
    def generate_points_at_inclination(n_samples: int, incl: float, r: float = 1) -> typing.List[Point]:
        if 180 - incl < 5:
            return [Point.pol(r, 0, 0)]
        elif 175 < 180 - incl:
            return [Point.pol(r, 180, 0)]
        n_azimuth_samples = [i * 360 / n_samples for i in range(n_samples + 1)]
        points = [Point.pol(r, incl, azim) for azim in n_azimuth_samples]
        return points

    @staticmethod
    def generate_points_at_inclinations(n_samples, inclinations: typing.List[float]) -> typing.List[Point]:
        points = [point for incl in inclinations for point in Points.generate_points_at_inclination(n_samples, incl)]
        return points

    @staticmethod
    def split_points_to_xyz_lists(points: typing.List[Point]) -> typing.Tuple[typing.List, ...]:
        x = [p.to_cart()[0] for p in points]
        y = [p.to_cart()[1] for p in points]
        z = [p.to_cart()[2] for p in points]
        return x, y, z

    @staticmethod
    def build_kd_tree(points: typing.List[Point]) -> KDTree:
        coordinates = [p.to_cart() for p in points]
        kd_tree = KDTree(coordinates)
        return kd_tree

    @staticmethod
    def convert_rays_to_points(rays, kd_tree, max_distance):
        points = []
        for ray in rays:
            origin, direction = ray

            # Perform ray marching until the maximum distance is reached or an intersection occurs
            t = 0.0
            while t < max_distance:
                point = origin + t * direction

                # Query the KD tree for the closest point
                dist, idx = kd_tree.query(point)

                # Check if the distance to the closest point is within a tolerance
                if dist < 0.001:
                    points.append(point)
                    break

                t += dist

        return points


def plot_point_3d():
    points = Points.generate_points_at_inclinations(10, [i for i in range(0, 191, 30)])
    x, y, z = Points.split_points_to_xyz_lists(points)

    # Build the kd-tree
    kd_tree = Points.build_kd_tree(points)

    # Store each node and its immediate neighbors
    nodes_with_neighbors = []
    for i, point in enumerate(points):
        # Convert the current point to a list with shape (3,)
        current_point = [x[i], y[i], z[i]]

        # Find the immediate neighbors for the current point
        neighbors = kd_tree.query_ball_point(current_point, r=0.1)

        # Create a node and store its point and neighbors
        node = {
            'point': point,
            'neighbors': [points[j] for j in neighbors if j != i]
        }
        nodes_with_neighbors.append(node)

        # Generate rays
        rays = []
        for i in range(len(points)):
            origin = np.array([x[i], y[i], z[i]])
            direction = np.array([0.0, 0.0, 1.0])  # Example direction, modify as needed
            rays.append((origin, direction))

        # Convert rays to 3D points using the KD tree
        max_distance = 10.0  # Maximum distance to march along each ray
        converted_points = Points.convert_rays_to_points(rays, kd_tree, max_distance)

        for point in converted_points:
            print(point)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z, mode="markers", marker=dict(size=5, color="red"),
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="X",
                nticks=4,
                range=[-1, 1],
                backgroundcolor="rgb(255, 255, 255)",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="gray",
            ),
            yaxis=dict(
                title="Y",
                nticks=4,
                range=[-1, 1],
                backgroundcolor="rgb(255, 255, 255)",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="gray",
            ),
            zaxis=dict(
                title="Z",
                nticks=4,
                range=[-1, 1],
                backgroundcolor="rgb(255, 255, 255)",
                gridcolor="gray",
                showbackground=True,
                zerolinecolor="gray",
            ),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
        )
    )

    fig.show()


plot_point_3d()
