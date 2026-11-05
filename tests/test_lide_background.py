import numpy as np
from scipy.linalg import expm

from mylib.background_cosmodels import LIDE


def _matrix_solution(model, a):
    matrix = np.array([
        [model.k2 - 1.0, model.k1],
        [-model.k2, -(1.0 + model.w + model.k1)],
    ])
    initial = np.array([model.rhodm0, model.rhol0])
    return expm(3.0*np.log(a)*matrix) @ initial


def _assert_matches_matrix_exponential(model):
    assert np.isclose(model.Rho_dm(1.0), model.rhodm0, rtol=2e-13)
    assert np.isclose(model.Rho_L(1.0), model.rhol0, rtol=2e-13)
    for a in (0.02, 0.2, 0.7, 1.0, 1.4):
        expected = _matrix_solution(model, a)
        actual = np.array([model.Rho_dm(a), model.Rho_L(a)])
        assert np.allclose(actual, expected, rtol=2e-11, atol=1e-8)


def test_real_mode_branch():
    _assert_matches_matrix_exponential(LIDE(Ksi_1=0.01, Ksi_2=-0.005, omega=-1.0))


def test_degenerate_branch():
    model = LIDE(Ksi_1=0.25, Ksi_2=0.25, omega=-1.0)
    assert model.DeltaSym == 0
    _assert_matches_matrix_exponential(model)


def test_oscillatory_branch():
    model = LIDE(Ksi_1=0.30, Ksi_2=0.30, omega=-1.0)
    assert model.DeltaSym < 0
    _assert_matches_matrix_exponential(model)


def test_xi2_zero_regular_case():
    _assert_matches_matrix_exponential(LIDE(Ksi_1=0.08, Ksi_2=0.0, omega=-1.0))


def test_xi2_zero_degenerate_case():
    model = LIDE(Ksi_1=1.0, Ksi_2=0.0, omega=-1.0)
    assert model.DeltaSym == 0
    _assert_matches_matrix_exponential(model)


def test_lcdm_limit():
    model = LIDE(Ksi_1=0.0, Ksi_2=0.0, omega=-1.0)
    for a in (0.01, 0.1, 0.5, 1.0):
        assert np.isclose(model.Rho_dm(a), model.rhodm0*a**-3, rtol=2e-13)
        assert np.isclose(model.Rho_L(a), model.rhol0, rtol=2e-13)
