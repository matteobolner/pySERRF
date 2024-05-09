"""
Various functions for pyserrf.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def read_and_parse_excel(input_path):
    """
    Read data in serrf compatible format from an Excel file
    prepared as specified in the original SERRF R package documentation.

    Parameters
    ----------
    input_path : str
        Path to the input file.

    Returns
    -------
    data : pandas.DataFrame
        A pandas DataFrame containing the data in the Excel file.
    """
    data = pd.read_excel(input_path, header=None)
    data.replace("", np.nan, inplace=True)

    # Transpose the data to get it in the correct format
    data = data.transpose()

    # Use the second row as the column names
    data.columns = data.iloc[1]

    # Drop the first two rows of the data
    data = data.iloc[2::]

    # Remove the index created by the transpose
    data.columns.name = None
    data = data.reset_index(drop=True)

    # Reset the index and return the data
    return data


def center_data(data: np.ndarray) -> np.ndarray:
    """
    Center the data by subtracting the mean.

    Parameters
    ----------
    data : array-like
        The data to be centered.

    Returns
    -------
    centered : array-like
        The centered data.
    """
    mean = np.nanmean(data)
    centered = data - mean
    return centered


def standard_scaler(data: pd.DataFrame) -> pd.DataFrame:
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
    scaler = StandardScaler()
    scaler.fit(data)
    scaler.scale_ = np.std(data, axis=0, ddof=1).replace(0, 1).to_list()
    scaled = scaler.transform(data)
    scaled = pd.DataFrame(scaled, columns=data.columns, index=data.index)
    return scaled


def scale_columns(data, columns):
    """
    Scales the data for group of columns.

    Parameters
    ----------
    data : pandas DataFrame
        The data containing the columns to be scaled
    columns : iterable
        The names of the columns to scale

    Returns
    -------
    pandas DataFrame
        The scaled data for the given columns.
    """
    data = data[list(columns)]
    if data.empty:
        return pd.DataFrame()
    return standard_scaler(data)


def sorted_intersection(list1, list2):
    """
    Return the intersection of two lists sorted by their appearance in
    list1.
    """
    return [i for i in list1 if i in list2]


def get_top_values_in_both_correlations(series1, series2, num):
    """
    Get the index of the top {num} metabolites that are in both of the given correlations.

    Parameters
    ----------
    series1 : pandas.Series
        The first correlation series.
    series2 : pandas.Series
        The second correlation series.
    num : int
        The number of metabolites to select.

    Returns
    -------
    set
        The names of the top metabolites that are in both correlations.

    Notes
    -----
    This function selects the top `num` metabolites that appear in both
    `series1` and `series2`. It does this by iteratively increasing the
    number of metabolites selected until it has reached `num` metabolites.
    """

    selected = []
    for i in range(len(series1)):
        if len(selected) < num:
            templist_1 = list(series1.iloc[0:i].index)
            templist_2 = list(series2.iloc[0:i].index)
            temp_intersection = sorted_intersection(templist_1, templist_2)
            for k in temp_intersection:
                if k not in selected:
                    selected.append(k)
                    if len(selected) == num:
                        break
        else:
            break
    selected = selected[0:num]
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


def get_sorted_correlation(correlation_matrix, metabolite):
    """
    Returns a sorted Series of absolute correlations for the given metabolite,
    sorted from highest to lowest. The given metabolite is dropped from the
    resulting Series.

    Parameters
    ----------
    correlation_matrix : pandas DataFrame
        The correlation matrix of the data.
    metabolite : str
        The name of the metabolite for which the sorted correlations are desired.

    Returns
    -------
    pandas Series
        The sorted correlations for the given metabolite.
    """

    sorted_correlation = (
        correlation_matrix.abs()  # get absolute values
        .loc[metabolite]  # restrict to rows for the given metabolite
        .sort_values(ascending=False)  # sort from highest to lowest
        .drop(metabolite)  # drop the given metabolite (now in index)
    )

    return sorted_correlation


def predict_values(qc_data_x, qc_data_y, target_data_x, random_state=None):
    """
    Predicts the values of the target data using a random forest regressor.

    The qc data is used to build a random forest regressor, which is then
    used to predict the values of the target data. The predicted values are returned
    as a tuple with the qc prediction and the target prediction.

    Parameters
    ----------
    qc_data_x : pandas DataFrame
        The qc data features.
    qc_data_y : pandas Series
        The qc data targets.
    target_data_x : pandas DataFrame
        The target data features.

    Returns
    -------
    tuple
        A tuple containing the qc prediction and the target prediction.
    """
    model = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, random_state=random_state
    )
    model.fit(X=qc_data_x, y=qc_data_y, sample_weight=None)
    qc_prediction = model.predict(qc_data_x)
    target_prediction = model.predict(target_data_x)
    return qc_prediction, target_prediction
