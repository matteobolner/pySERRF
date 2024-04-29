import pandas as pd
import numpy as np
import pytest
from pyserrf.utils import replace_zero_values, replace_nan_values, standard_scaler


def test_replace_multiple_zero_values(monkeypatch):
    """
    Test that replace_zero_values replaces multiple zero values in a row
    """
    row = pd.Series([0, 0, 0, 5, 0, 0])
    replaced = replace_zero_values(row)
    assert len(replaced[replaced == 0]) == 0


def test_no_zero_values(monkeypatch):
    """
    Test that replace_zero_values returns the input row unchanged if there are
    no zero values.
    """
    row = pd.Series([1, 2, 3, 4, 5])
    expected_output = pd.Series([1, 2, 3, 4, 5])
    assert replace_zero_values(row).equals(
        expected_output
    ), "Should not modify input row if there are no zero values"
