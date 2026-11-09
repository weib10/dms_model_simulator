"""Pinhole camera with optional radial distortion, placed in the vehicle frame."""
from __future__ import annotations
import numpy as np
from . import geometry as G


class Camera:
    def __init__(self, name: str, spec: dict, cabin_head_center: np.ndarray):
        self.name = name
        self.position = np.array(spec["position"], float)
        self.width = int(spec["width"])
        self.height = int(spec["height"])
        self.hfov = float(spec["hfov_deg"])
        self.k1, self.k2 = [float(x) for x in spec.get("distortion", [0, 0])]
        self.fx = (self.width / 2) / np.tan(np.radians(self.hfov) / 2)
        self.fy = self.fx
        self.cx, self.cy = self.width / 2, self.height / 2
        orient = spec.get("orientation", "look_at_head")
        if orient == "look_at_head":
            target = cabin_head_center + np.array(spec.get("look_at_offset", [0, 0, 0]), float)
        else:
            d = G.yaw_pitch_to_dir(float(orient["yaw"]), float(orient["pitch"]))
            target = self.position + d
        self.R = G.look_at(self.position, target)          # cam_from_vehicle
        # mounting error: the camera sits mount_offset (vehicle frame, m) away from its design position, keeping
        # the design aim, and is then rotated about its own axes by mount_rotation_deg [yaw, pitch, roll]:
        # yaw > 0 pans the optical axis towards image right, pitch > 0 tilts it towards image top, roll > 0
        # turns the camera clockwise about the axis as seen from behind it (the scene turns anticlockwise)
        self.mount_offset = np.array(spec.get("mount_offset", [0.0, 0.0, 0.0]), float)
        self.mount_rotation = np.array(spec.get("mount_rotation_deg", [0.0, 0.0, 0.0]), float)
        self.position = self.position + self.mount_offset
        if np.any(self.mount_rotation):
            self.R = mount_rotation(*self.mount_rotation).T @ self.R
        self.optical_axis = self.R[2]                      # vehicle frame
        self.vfov = 2 * np.degrees(np.arctan((self.height / 2) / self.fy))
        self.r2_max = self._fold_r2(self.k1, self.k2)

    @staticmethod
    def _fold_r2(k1: float, k2: float) -> float:
        """Squared normalised radius where r_d = r (1 + k1 r^2 + k2 r^4) stops increasing: first positive root of
        d r_d / d r = 1 + 3 k1 r^2 + 5 k2 r^4. Beyond it the model folds back into the image (inf = never)."""
        roots = np.roots([5 * k2, 3 * k1, 1.0]) if k2 != 0 else (np.array([-1.0 / (3 * k1)]) if k1 != 0 else [])
        pos = [float(r.real) for r in np.atleast_1d(roots) if abs(r.imag) < 1e-12 and r.real > 0]
        return min(pos) if pos else np.inf

    def _distortion(self, pc: np.ndarray) -> tuple[float, float, float] | None:
        """(x, y, radial scale) of a camera-frame point, None if behind the near plane or past the fold radius."""
        if pc[2] <= 0.05:
            return None
        x, y = pc[0] / pc[2], pc[1] / pc[2]
        r2 = x * x + y * y
        if r2 >= self.r2_max:
            return None
        return x, y, 1 + self.k1 * r2 + self.k2 * r2 * r2

    @classmethod
    def from_config(cls, name: str, cams_cfg: dict, cabin_head_center: np.ndarray) -> "Camera":
        spec = {**cams_cfg.get("defaults", {}), **cams_cfg["cameras"][name]}
        return cls(name, spec, cabin_head_center)

    # ---- transforms -----------------------------------------------------------
    def to_cam(self, p_vehicle: np.ndarray) -> np.ndarray:
        return self.R @ (np.asarray(p_vehicle, float) - self.position)

    def dir_to_cam(self, d_vehicle: np.ndarray) -> np.ndarray:
        return self.R @ np.asarray(d_vehicle, float)

    def dir_to_vehicle(self, d_cam: np.ndarray) -> np.ndarray:
        return self.R.T @ np.asarray(d_cam, float)

    def project(self, p_vehicle: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (u, v) pixel and a flag for 'in front of camera and inside the image'."""
        dist = self._distortion(self.to_cam(p_vehicle))
        if dist is None:
            return np.array([np.nan, np.nan]), False
        x, y, k = dist
        u, v = self.fx * x * k + self.cx, self.fy * y * k + self.cy
        inside = 0 <= u < self.width and 0 <= v < self.height
        return np.array([u, v]), inside

    def project_size(self, p_vehicle: np.ndarray, size_m: float) -> float:
        """Approximate pixel size of an object of metric size at that depth (same radial distortion scale as
        project; 0 where project rejects the point)."""
        pc = self.to_cam(p_vehicle)
        dist = self._distortion(pc)
        return float(self.fx * size_m / pc[2] * dist[2]) if dist is not None else 0.0

    def describe(self) -> dict:
        return {
            "name": self.name, "position": self.position.tolist(),
            "width": self.width, "height": self.height, "hfov_deg": self.hfov,
            "vfov_deg": float(self.vfov), "fx": float(self.fx), "distortion": [self.k1, self.k2],
            "R_cam_from_vehicle": self.R.tolist(), "optical_axis_vehicle": self.optical_axis.tolist(),
            "mount_offset": self.mount_offset.tolist(), "mount_rotation_deg": self.mount_rotation.tolist(),
        }


def mount_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Axes of the rotated camera in the design camera frame (OpenCV: x right, y down, z forward), as columns.
    yaw turns z towards +x (image right), pitch towards -y (image top), roll turns x towards +y."""
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz
