import pandas as pd
import numpy as np


def replace_zero_values(row: pd.Series) -> pd.Series:
    """
    Replace zero values in a row of a pandas DataFrame with a normally distributed
    random variable. The normal distribution has a mean of the minimum non-NaN value
    in the row plus 1, and a standard deviation of 10% of the minimum non-NaN value.

    Parameters
    ----------
    row : pandas.Series
        A row of a pandas DataFrame

    Returns
    -------
    pandas.Series
        The input row with zero values replaced by normally distributed random
        variables
    """
    zero_values = row[row == 0].index  # Indices of zero values
    min_non_nan = row.dropna().min()  # Minimum non-NaN value in the row
    mean = min_non_nan + 1  # Mean of the normal distribution
    std = 0.1 * (min_non_nan + 0.1)  # Standard deviation of the normal distribution
    zero_replacements = np.random.normal(
        loc=mean,
        scale=std,
        size=len(zero_values),
    )
    row.loc[zero_values] = zero_replacements
    return row


def replace_nan_values(row):
    """
    Replace NaN values in a row of a pandas DataFrame with normally distributed
    random variables. The normal distribution has a mean of half the minimum
    non-NaN value in the row plus one, and a standard deviation of 10% of the
    minimum non-NaN value.

    Parameters
    ----------
    row : pandas.Series
        A row of a pandas DataFrame

    Returns
    -------
    pandas.Series
        The input row with NaN values replaced by normally distributed random
        variables
    """
    nan_values = row[row.isna()]  # Indices of NaN values
    non_nan_values = row.dropna()  # Non-NaN values in the row
    min_non_nan = np.min(non_nan_values)  # Minimum non-NaN value in the row
    mean = 0.5 * (min_non_nan + 1)  # Mean of the normal distribution
    std = 0.1 * (min_non_nan + 0.1)  # Standard deviation of the normal distribution
    nan_replacements = np.random.normal(
        loc=mean,
        scale=std,
        size=len(nan_values),
    )  # Random variables
    row.loc[nan_values.index] = nan_replacements  # Replace NaN values
    return row  # Return the row with replaced NaN values

def center_data(data: np.ndarray) -> np.ndarray:
    mean = data.mean()
    centered = data - mean
    return centered

def standard_scaler(data: np.ndarray) -> np.ndarray:
    """
    Standardize data by subtracting the mean and dividing by the standard deviation.

    Parameters
    ----------
    data : array-like
        The data to be standardized.

    Returns
    -------
    array-like
        The standardized data.
    """
    centered=center_data(data)
    std = data.std(ddof=1)
    scaled = centered / std
    return scaled
