"""Test the reference threshold of findRigidTransformation."""
import os

import numpy as np
import pymagsac

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(THIS_PATH, "..", "graph-cut-ransac/build/data/rigid_pose_example")
CORRESPONDENCES = np.loadtxt(os.path.join(DATA, "rigid_pose_example_points.txt"))
GT_POSE = np.loadtxt(os.path.join(DATA, "rigid_pose_example_gt.txt"))[:4, :]
THRESHOLD = 0.03  # metres


def rotation_error(pose):
    """Rotation error of an estimated pose (as returned, transposed), in degrees."""
    R = GT_POSE[:3, :3].T @ pose.T[:3, :3]
    return np.degrees(np.arccos(np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)))


def estimate_rigid(scale):
    return pymagsac.findRigidTransformation(
        np.ascontiguousarray(CORRESPONDENCES * scale),
        probabilities=[],
        sampler=0,
        sigma_th=THRESHOLD * scale,
        min_iters=50,
        max_iters=5000,
    )


def test_rigid_transformation_in_millimetres_as_in_metres():
    # The reference threshold of findRigidTransformation is sigma_th, so the
    # same data give a model in millimetres as in metres (with a fixed 1.0, in
    # millimetres, no model was found).
    pose_m, mask_m = estimate_rigid(1.0)
    pose_mm, mask_mm = estimate_rigid(1000.0)
    assert pose_m is not None and pose_mm is not None
    assert rotation_error(pose_m) < 4  # degrees
    assert rotation_error(pose_mm) < 4
    assert abs(int(mask_mm.sum()) - int(mask_m.sum())) < 0.2 * mask_m.sum()
