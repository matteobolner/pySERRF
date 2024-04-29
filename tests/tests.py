import numpy as np
import pandas as pd
import pytest
from pyserrf import SERRF
from sklearn.preprocessing import StandardScaler


class TestCenterData(object):
    @pytest.fixture(autouse=True)
    def serrf(self):
        self.serrf = SERRF()

    def test_returns_same_shape(self):
        data = np.array([1, 2, 3])
        result = self.serrf._center_data(data)
        assert result.shape == data.shape

    def test_returns_same_values_when_mean_is_zero(self):
        data = np.array([-1, 0, 1])
        result = self.serrf._center_data(data)
        np.testing.assert_array_equal(result, data)

    def test_returns_shifted_values_when_mean_is_not_zero(self):
        data = np.array([1, 2, 3])
        mean = np.mean(data)
        expected_result = data - mean
        result = self.serrf._center_data(data)
        np.testing.assert_array_equal(result, expected_result)


class TestStandardScaler:
    @pytest.fixture
    def serrf(self):
        return SERRF()

    def test_standard_scaler_returns_correct_values(self, serrf):
        data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        expected_result = pd.DataFrame(
            [[-0.71, -0.71], [0.71, 0.71]], columns=["A", "B"]
        )
        result = serrf._standard_scaler(data).round(2)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_standard_scaler_handles_zero_std(self, serrf):
        data = pd.DataFrame([[1, 1], [1, 1]], columns=["A", "B"])
        expected_result = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], columns=["A", "B"])
        result = serrf._standard_scaler(data)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_standard_scaler_handles_nan_values(self, serrf):
        data = pd.DataFrame([[3, np.nan], [2, 5], [4, 3]], columns=["A", "B"])
        expected_result = pd.DataFrame(
            [[0.0, np.nan], [-1, 0.71], [1, -0.71]], columns=["A", "B"]
        )
        result = serrf._standard_scaler(data).round(2)
        pd.testing.assert_frame_equal(result, expected_result)
