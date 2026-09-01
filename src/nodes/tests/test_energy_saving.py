import numpy as np
import pytest

from nodes.actions.energy_saving import BuildingEnergyParams, simulate_building_energy_saving

pytestmark = pytest.mark.django_db


def test_simulate_building_energy_saving_reaches_plateau_without_reinvestment_cost():
    params = BuildingEnergyParams(
        start_year=2024,
        nr_years=31,
        lifetime=10,
        renovation_rate=0.1,
        renovation_potential=0.5,
        investment_cost=100.0,
        maint_cost=2.0,
        he_saving=5.0,
        el_saving=1.0,
        all_in_investment=False,
    )

    ret = simulate_building_energy_saving(params)

    np.testing.assert_allclose(ret.total_renovated[:6], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(ret.total_renovated[6:16], np.full(10, 0.5))
    np.testing.assert_array_equal(ret.forecast[:3], [0, 1, 1])

    # The current production logic keeps charging only maintenance once the stock
    # is saturated, even after renovations start expiring.
    np.testing.assert_allclose(ret.cost[11:16], np.full(5, 1.0))
    np.testing.assert_allclose(ret.he_saving[11:16], np.full(5, -2.5))
    np.testing.assert_allclose(ret.el_saving[11:16], np.full(5, -0.5))
