# -----------------------------------------------------------------------------
# Copyright (c) 2025 Simon Josef Klein
#
# This work is licensed under the MIT License.
# See the LICENSE file in this repository or visit:
# https://opensource.org/licenses/MIT
#
# Author: Simon Klein
# Date: 2025
# Source: https://github.com/kleinsimon/labview-variant-data
# -----------------------------------------------------------------------------

import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import pytest

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from labview_data import serialize_variant, deserialize_variant, NamedItem

@pytest.fixture
def items():
    return [
        {1,3,5,4},
        100,
        100.1,
        True,
        False,
        "Hallo Welt!",
        Path(r"C:\Windows"),
        Path(r"..\Windows"),
        np.arange(5, dtype=np.uint8),
        ["Test", "Hallo Welt"],
        ("a", "bc", ("x", "y", "de")),
        (np.int32(1), np.int32(2), 3.5, "Mieze"),
        datetime.now(tz=timezone.utc),
        {"Hallo": np.int32(3), "Welt": np.int32(4)},
        {1: "test", "2": 5},
        None,
        {1: None, "banane": 55},
        np.linspace(1, 10, 20)
    ]


def test_serialize_variant(items):
    for v in items:
        buffer = serialize_variant(v)
        print(v, buffer.hex())


def test_deserialize_variant(items):
    for v in items:
        buffer = serialize_variant(v)
        print(v, buffer.hex())
        v2 = deserialize_variant(buffer)
        print(v, v2)

        if isinstance(v, np.ndarray):
            v = list(v)

        if isinstance(v2, np.ndarray):
            v2 = list(v2)

        assert v == v2, f"Value mismatch for serialization of {v}"

