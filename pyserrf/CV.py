"""
This module contains the cross_validate function, which performs cross-validation
of the SERRF algorithm using the QC samples.
"""

import numpy as np
import pandas as pd
from scipy.stats import variation
from sklearn.model_selection import StratifiedKFold
from pyserrf import SERRF


def cross_validate(
    dataset,
    n_splits=5,
    sample_type_column="sampleType",
    batch_column="batch",
    time_column="time",
    other_columns=None,
    n_correlated_metabolites=10,
    random_state=None,
    threads=1,
):
    """Perform cross-validation of the SERRF algorithm using the QC samples

    Parameters
    ----------
    dataset : Pandas DataFrame
        The SERRF-compatible dataset.
    n_splits : int, optional
        The number of CV folds to use, by default 5
    sample_type_column : str, optional
        The name of the column in the dataset that contains the sample type
        information, by default "sampleType"
    batch_column : str, optional
        The name of the column in the dataset that contains the batch
        information, by default "batch"
    time_column : str, optional
        The name of the column in the dataset that contains the injection time
        information, by default "time"
    other_columns : list, optional
        A list of additional metadata columns to exclude from the data, by default None
    n_correlated_metabolites : int, optional
        The number of correlated metabolites to use for the normalization, by default 10
    random_state : int, optional
        The random state, by default None
    threads : int, optional
        The number of threads, by default 1

    Returns
    -------
    raw_variation : Pandas DataFrame
        The coefficient of variation of each metabolite in the raw data
    normalized_variation : Pandas DataFrame
        The coefficient of variation of the normalized data
    """
    raw_variations = []
    normalized_variations = []
    # Filter out QC samples
    dataset = dataset[dataset[sample_type_column] == "qc"]
    dataset = dataset.reset_index(drop=True)
    # Add a batch column if it is None
    if batch_column is None:
        dataset["batch"] = 1
        batch_column = "batch"
    # Get the batch labels
    y = dataset[batch_column]
    # Create a StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    # Perform CV
    for i, (train_index, test_index) in enumerate(skf.split(dataset, y)):
        print(f"CV {i}/{n_splits}")
        # Create training and target dataframes
        temp_qc = dataset.loc[train_index]
        temp_target = dataset.loc[test_index]
        # Change the target type to "sample"
        temp_target[sample_type_column] = "sample"
        # Concatenate the two dataframes
        tempdf = pd.concat([temp_qc, temp_target])
        # Sort the data by time if time_column is not None
        if time_column is not None:
            tempdf = tempdf.sort_values(by=time_column)
        # Create a SERRF object
        tempserrf = SERRF(
            sample_type_column=sample_type_column,
            batch_column=batch_column,
            time_column=time_column,
            other_columns=other_columns,
            random_state=random_state,
            n_correlated_metabolites=n_correlated_metabolites,
            threads=threads,
        )
        # Fit and transform the data
        normalized = tempserrf.fit_transform(tempdf)
        # Select the normalized target samples
        normalized_target = normalized[normalized["sampleType"] == "sample"]
        # Select the raw target samples
        raw_target = temp_target
        # Compute the RSD of the raw target samples
        raw_target_variation = raw_target[tempserrf._metabolite_names].apply(variation)
        # Compute the RSD of the normalized target samples
        normalized_target_variation = normalized_target[
            tempserrf._metabolite_names
        ].apply(variation)
        # Add the index to the RSD DataFrame
        raw_target_variation.name = f"{i}"
        normalized_target_variation.name = f"{i}"
        # Append the RSD data to the list
        raw_variations.append(raw_target_variation)
        normalized_variations.append(normalized_target_variation)
    # Concatenate the RSD dataframes
    raw_variations = pd.concat(raw_variations, axis=1)
    normalized_variations = pd.concat(normalized_variations, axis=1)
    return raw_variations, normalized_variations
