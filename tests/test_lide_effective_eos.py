import numpy as np
import pytest

from mylib import LIDE


@pytest.mark.parametrize('k1,k2', [(0.01, -0.005), (0.25, 0.25), (0.3, 0.3)])
@pytest.mark.parametrize('omega', [-1.0, -0.9])
def test_effective_eos_matches_density_evolution(k1, k2, omega):
    model = LIDE(Ksi_1=k1, Ksi_2=k2, omega=omega)
    a = np.array([0.6, 0.8, 1.0, 1.2])
    step = 1e-5

    def derivative(density):
        return (density(a*np.exp(step)) - density(a*np.exp(-step)))/(2*step)

    for density, eos in [(model.Rho_dm, model.w_eff_dm),
                         (model.Rho_L, model.w_eff_L)]:
        expected = -1 - derivative(density)/(3*density(a))
        np.testing.assert_allclose(eos(a), expected, rtol=2e-6, atol=2e-8)

    dm0 = 0.27

    def reconstructed_density(scale):
        return (model.Rho_dm(scale) + model.Rho_L(scale)
                - dm0*model.rho_crit*scale**(-3))

    expected = -1 - derivative(reconstructed_density)/(3*reconstructed_density(a))
    np.testing.assert_allclose(model.w_eff_rec(a, dm0=dm0), expected,
                               rtol=2e-6, atol=2e-8)


def test_uncoupled_limit_and_reference_normalization():
    model = LIDE(Ksi_1=0.0, Ksi_2=0.0, omega=-1.0)
    a = np.array([0.2, 0.5, 1.0, 1.3])
    np.testing.assert_allclose(model.w_eff_dm(a), 0.0)
    np.testing.assert_allclose(model.w_eff_L(a), -1.0)
    np.testing.assert_allclose(model.w_eff_rec(a, dm0=model.Omega_dm0), -1.0)
    expected = -model.rhol0/(model.rhol0
                            + (model.rhodm0 - 0.3*model.rho_crit)*a**(-3))
    np.testing.assert_allclose(model.w_eff_rec(a), expected)


def test_scalar_and_list_inputs():
    model = LIDE()
    for eos in [model.w_eff_dm, model.w_eff_L, model.w_eff_rec]:
        assert np.ndim(eos(1.0)) == 0
        np.testing.assert_allclose(eos([0.5, 1.0]),
                                   [eos(0.5), eos(1.0)])
