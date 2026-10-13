# ruff: noqa: D100, D103
from typing import Literal

import numpy as np
import pytest

from risb.kweight import SmearingKWeight

energies = {
    "up": np.linspace(-1, 1, 100).reshape(100, 1),
    "dn": np.linspace(-1, 1, 100).reshape(100, 1),
}


@pytest.mark.parametrize("method", ["fermi", "gaussian", "methfessel-paxton"])
def test_weights_sum_to_n_target(
    method: Literal["fermi", "gaussian", "methfessel-paxton"],
) -> None:
    kweight = SmearingKWeight(beta=10, n_target=1.2, method=method)
    weights = kweight.update_weights(energies)
    assert sum(w.sum() for w in weights.values()) == pytest.approx(1.2)


def test_update_weights_without_mu_or_n_target_raises() -> None:
    kweight = SmearingKWeight(beta=10)
    with pytest.raises(ValueError, match="mu or n_target"):
        kweight.update_weights(energies)
