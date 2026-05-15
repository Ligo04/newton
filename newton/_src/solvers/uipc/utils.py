from __future__ import annotations

from typing import Any

import numpy as np
from uipc import view


def _view_attr(slot: Any) -> np.ndarray:
    return view(slot)  # pyright: ignore[reportArgumentType]
