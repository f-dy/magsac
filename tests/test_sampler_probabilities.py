"""Samplers 3 (importance) and 4 (AR-Sampler) need one probability per point, finite and non-negative,
at least one positive."""
import numpy as np
import pymagsac
import pytest

N = 50
DATA = np.ascontiguousarray(np.random.default_rng(0).uniform(0, 100, (N, 4)))
K = np.eye(3)

FUNCTIONS = {
    "findHomography": lambda p, s: pymagsac.findHomography(
        DATA, 100, 100, 100, 100, probabilities=p, sampler=s),
    "findFundamentalMatrix": lambda p, s: pymagsac.findFundamentalMatrix(
        DATA, 100, 100, 100, 100, probabilities=p, sampler=s),
    "findEssentialMatrix": lambda p, s: pymagsac.findEssentialMatrix(
        DATA, K, K, 100, 100, 100, 100, probabilities=p, sampler=s),
    "findLine2D": lambda p, s: pymagsac.findLine2D(
        np.ascontiguousarray(DATA[:, :2]), 100, 100, probabilities=p, sampler=s),
    "findRigidTransformation": lambda p, s: pymagsac.findRigidTransformation(
        np.ascontiguousarray(np.c_[DATA, DATA[:, :2]]), probabilities=p, sampler=s),
}


@pytest.mark.parametrize("name", FUNCTIONS)
def test_no_probabilities(name):
    # An empty array crashed, or gave no model, without an error.
    with pytest.raises(ValueError, match="one probability per point"):
        FUNCTIONS[name]([], 4)


@pytest.mark.parametrize("probabilities, message", [
    (np.zeros(N), "positive probability"),
    (np.r_[-0.5, np.ones(N - 1)], "non-negative"),
    (np.r_[np.nan, np.ones(N - 1)], "non-negative")])
def test_invalid_probabilities(probabilities, message):
    with pytest.raises(ValueError, match=message):
        FUNCTIONS["findHomography"](probabilities, 3)


def test_one_probability_per_point():
    FUNCTIONS["findHomography"](np.ones(N), 4)
