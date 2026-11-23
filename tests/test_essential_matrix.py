"""Test that ``pymagsac.findEssentialMatrix`` returns its refined model.

The function ends with a bundle adjustment of the relative pose that minimizes
the squared Sampson errors of the inliers, so a Gauss-Newton step from the
returned essential matrix should barely decrease that cost. From the unrefined
model, it decreases by about 0.01% to 0.1%.
"""
import numpy as np
import pymagsac


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _rotation(w):
    angle = np.linalg.norm(w)
    if angle == 0:
        return np.eye(3)
    k = _skew(w / angle)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _sampson(E, x1, x2):
    """Sampson errors of the matches x1 <-> x2 (homogeneous rows)."""
    Ex1 = x1 @ E.T
    Etx2 = x2 @ E
    num = np.sum(x2 * Ex1, axis=1)
    return num / np.sqrt(Ex1[:, 0]**2 + Ex1[:, 1]**2 + Etx2[:, 0]**2 + Etx2[:, 1]**2)


def _relative_cost_decrease(E, x1, x2):
    """Relative decrease of the sum of squared Sampson errors by one Gauss-Newton step."""
    U, _, Vt = np.linalg.svd(E)

    def essential(p):
        # E = U diag(1, 1, 0) Vt, with U and V rotated (5 degrees of freedom)
        return U @ _rotation(p[:3]) @ np.diag([1.0, 1.0, 0.0]) @ (Vt.T @ _rotation(np.r_[p[3:], -p[2]])).T

    r = _sampson(E, x1, x2)
    J = np.array([(_sampson(essential(1e-7 * dp), x1, x2) - r) / 1e-7 for dp in np.eye(5)]).T
    step = np.linalg.lstsq(J, -r, rcond=None)[0]
    cost0 = np.sum(r**2)
    return (cost0 - np.sum(_sampson(essential(step), x1, x2)**2)) / cost0


def test_returned_model_is_refined():
    rng = np.random.default_rng(0)
    K = np.array([[800.0, 0, 320], [0, 800, 240], [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    R = _rotation(np.radians(10.0) * np.array([0.6, 0.8, 0.0]))
    t = np.array([1.0, 0.2, 0.1])
    E_true = _skew(t) @ R
    n_inliers, n_outliers = 200, 100
    for _ in range(3):
        # Noisy matches between two 640x480 views, and outliers far from the epipolar lines
        X = np.c_[rng.uniform(-3, 3, (4 * n_inliers, 2)), rng.uniform(4, 10, 4 * n_inliers)]
        p1 = X @ K.T
        p2 = (X @ R.T + t) @ K.T
        p1 = p1[:, :2] / p1[:, 2:]
        p2 = p2[:, :2] / p2[:, 2:]
        visible = np.all((p1 >= 0) & (p1 < [640, 480]) & (p2 >= 0) & (p2 < [640, 480]), axis=1)
        inliers = np.c_[p1, p2][visible][:n_inliers] + rng.normal(0, 0.5, (n_inliers, 4))
        outliers = rng.uniform(0, 1, (20 * n_outliers, 4)) * [640, 480, 640, 480]
        far = np.abs(_sampson(E_true, np.c_[outliers[:, :2], np.ones(len(outliers))] @ K_inv.T,
                              np.c_[outliers[:, 2:], np.ones(len(outliers))] @ K_inv.T)) * 800 > 50
        correspondences = np.r_[inliers, outliers[far][:n_outliers]]

        E, mask = pymagsac.findEssentialMatrix(
            np.ascontiguousarray(correspondences), K, K, 640, 480, 640, 480,
            np.zeros(0), sampler=0, sigma_th=5.0)
        assert E is not None
        np.testing.assert_array_equal(mask, np.arange(len(mask)) < n_inliers)
        x1 = np.c_[inliers[:, :2], np.ones(n_inliers)] @ K_inv.T
        x2 = np.c_[inliers[:, 2:], np.ones(n_inliers)] @ K_inv.T
        assert _relative_cost_decrease(E, x1, x2) < 1e-5
