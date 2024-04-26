import pandas as pd
import numpy as np


def replace_zero_values(values: np.array, random_seed: int = 42) -> pd.Series:
    """
    Replace zero values in a pandas series with a normally distributed
    random variable. The normal distribution has a mean of the minimum non-NaN value
    in the values plus 1, and a standard deviation of 10% of the minimum non-NaN value.

    Parameters
    ----------
    values : np.array
        A numpy array

    Returns
    -------
    np.array
        The input array with zero values replaced by normally distributed random
        variables
    """

    rng = np.random.default_rng(seed=random_seed)
    np.random.seed(random_seed)
    # zero_values = row[row == 0].index  # Indices of zero values
    zero_values = np.where(row == 0)[0]
    if len(zero_values) == 0:
        return row
    min_val = np.nanmin(row)  # Minimum non-NaN value in the row=
    mean = min_val + 1  # Mean of the normal distribution
    std = 0.1 * (min_val + 0.1)  # Standard deviation of the normal distribution
    zero_replacements = rng.normal(
        loc=mean,
        scale=std,
        size=len(zero_values),
    )
    row[zero_values] = zero_replacements
    return row


def replace_nan_values(values, random_seed: int = 42) -> pd.Series:
    """
    Replace NaN values in a values of a numpy array with normally distributed
    random variables. The normal distribution has a mean of half the minimum
    non-NaN value in the values plus one, and a standard deviation of 10% of the
    minimum non-NaN value.

    Parameters
    ----------
    values : np.array
        A numpy array

    Returns
    -------
    np.array
        The input values with NaN values replaced by normally distributed random
        variables
    """
    rng = np.random.default_rng(seed=random_seed)
    np.random.seed(random_seed)
    nan_values = np.where(np.isnan(values))  # Indices of NaN values
    if len(nan_values) == 0:
        return values
    min_non_nan = np.nanmin(values)  # Minimum non-NaN value in the values
    mean = 0.5 * (min_non_nan + 1)  # Mean of the normal distribution
    std = 0.1 * (min_non_nan + 0.1)  # Standard deviation of the normal distribution
    nan_replacements = rng.normal(
        loc=mean,
        scale=std,
        size=len(nan_values),
    )  # Random variables
    values[nan_values] = nan_replacements  # Replace NaN values
    return values  # Return the row with replaced NaN values


def check_for_nan_or_zero(data):
    if (0 not in data.values) & (~data.isna().any().any()):
        return False
    else:
        return True


def handle_zero_and_nan(merged, metabolites):
    if check_for_nan_or_zero(merged[metabolites]):
        groups = []
        for name, group in merged.groupby(by="batch"):
            if check_for_nan_or_zero(group[metabolites]):
                group[metabolites] = group[metabolites].apply(replace_zero_values)
                group[metabolites] = group[metabolites].apply(replace_nan_values)
                groups.append(group)
            else:
                groups.append(group)
        merged = pd.concat(groups)
        # CHECK ORDER
        return merged
    else:
        return merged


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
    centered = center_data(data)
    std = data.std(ddof=1)
    scaled = centered / std
    return scaled


def get_corrs_by_sample_type_and_batch(merged, metabolites):
    corrs_train = {}
    corrs_target = {}

    for batch, group in merged.groupby(by="batch"):
        batch_training = group[group["sampleType"] == "qc"]
        batch_training_scaled = batch_training[metabolites].apply(standard_scaler)
        corrs_train[batch] = batch_training_scaled.corr(method="spearman")
        batch_target = group[group["sampleType"] != "qc"]
        if len(batch_target) == 0:
            corrs_target[batch] = np.nan
        else:
            batch_target_scaled = batch_target[metabolites].apply(standard_scaler)
            corrs_target[batch] = batch_target_scaled.corr(method="spearman")
    return corrs_train, corrs_target


def get_top_metabolites_in_both_correlations(series1, series2, num):
    selected = set()
    l = num
    while len(selected) < num:
        selected = selected.union(
            set(series1[0:l].index).intersection(set(series2[0:l].index))
        )
        l += 1
    return selected


def detect_outliers(data, threshold=3):
    """
    Detect outlier values in a single-column numerical array.

    Parameters:
    - data: single-column numerical array
    - threshold: a factor that determines the range from the IQR

    Returns:
    - Boolean array indicating outliers (True) and non-outliers (False)
    """
    if len(data) == 0:
        raise ValueError("Input is empty.")
    if isinstance(data, pd.DataFrame):
        raise TypeError("DataFrame input is not supported.")

    median = np.nanmedian(data)
    q1 = np.nanquantile(data, 0.25)
    q3 = np.nanquantile(data, 0.75)
    iqr = q3 - q1
    cutoff_lower = median - (threshold * iqr)
    cutoff_upper = median + (threshold * iqr)
    is_outlier = (data < cutoff_lower) | (data > cutoff_upper)
    return is_outlier
