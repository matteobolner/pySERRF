from multiprocessing import Pool
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
from tqdm import tqdm


def read_and_parse_excel(input_path):
    data = pd.read_excel(input_path, header=None)
    data.replace("", np.nan, inplace=True)
    data = data.transpose()
    data.columns = data.iloc[1]
    data = data.iloc[2::]
    data.columns.name = None
    data = data.reset_index(drop=True)
    return data


class SERRF:
    """
    This class implements the SERRF (Systematical Error Removal using Random Forest) method,
    which is a qc-based sample normalization method designed for large-scale
    untargeted metabolomics data.
    data. The method was developed by the Fan et al. in 2015 [1]_
    (see https://slfan2013.github.io/SERRF-online/).

    The class takes as input a pandas DataFrame containing metabolomic data and
    sample metadata, and outputs a pandas DataFrame with the normalized data.

    The class has the following parameters:

    - `sample_type_column` is the name of the column in the sample metadata
      with the sample type information (i.e qc or normal sample). The default
      value is 'sampleType'.
    - `batch_column` is the name of the column in the sample metadata with the
      batch information. If `None`, all samples are considered as part the same
      batch. The default value is `None`.
    - `sample_metadata_columns` is a list with the names of the columns in the
      sample metadata; it is important to specify all the metadata columns to
      separate them from the metabolite abundance values.
      The default value is ['sampleType', 'batch', 'label', 'time'].
    - `random_state` is the random seed used for all methods with a random
      component (i.e numpy normal distribution, sklearn random forest regressor).
      The default value is `None`, which means that a random seed is
      generated automatically. To obtain reproducible results, set a specific
      random seed.

    - `n_correlated_metabolites` is the number of metabolites with the highest
        correlation to the metabolite to be normalized. The default value is 10.

    Attributes:
    -----------
    _metabolites : list
        List with the names of the metabolites.
    _dataset : pandas DataFrame
        DataFrame with the metabolomic data and the sample metadata.
    _metabolite_dict : dict
        Dictionary with the mapping between the original column names and the
        new column names (MET_1, MET_2, etc.).
    corrs_qc : pandas DataFrame
        DataFrame with the Pearson correlation coefficients between the
        metabolites and the batch information.
    corrs_target : pandas DataFrame
        DataFrame with the Pearson correlation coefficients between the
        metabolites and the samples.
    normalized_data : pandas DataFrame
        DataFrame with the normalized data.
    normalized_dataset : pandas DataFrame
        DataFrame with the normalized data and the sample metadata.

    References
    ----------
    .. [1] Fan et al.:
        Systematic Error Removal using Random Forest (SERRF) for Normalizing
        Large-Scale Untargeted Lipidomics Data
        Analytical Chemistry DOI: 10.1021/acs.analchem.8b05592
        https://slfan2013.github.io/SERRF-online/
    """

    def __init__(
        self,
        sample_type_column="sampleType",
        batch_column=None,
        sample_metadata_columns=("sampleType", "batch", "label", "time"),
        random_state=None,
        n_correlated_metabolites=10,
        threads=1,
    ):
        """
        Initialize the class.

        Parameters
        ----------
        sample_type_column : str, optional
            The name of the column in the sample metadata with the sample type
            information (i.e qc or normal sample). The default value is
            'sampleType'.
        batch_column : str or None, optional
            The name of the column in the sample metadata with the batch
            information. If None, all samples are considered as part the same
            batch. The default value is None.
        sample_metadata_columns : list of str, optional
            A list with the names of the columns in the sample metadata; it is
            important to specify all the metadata columns to separate them from
            the metabolite abundance values. The default value is
            ['sampleType', 'batch', 'label', 'time'].
        random_state : int, RandomState instance, or None, optional
            The random seed used for all methods with a random component (i.e
            numpy normal distribution, sklearn random forest regressor). The
            default value is None, which means that a random seed is generated
            automatically. To obtain reproducible results, set a specific random
            seed.
        n_correlated_metabolites : int, optional
            The number of metabolites with the highest correlation to the
            metabolite to be normalized. The default value is 10.

        Attributes
        ----------
        sample_metadata_columns : list of str
            The list of columns in the sample metadata.
        sample_type_column : str
            The name of the column in the sample metadata with the sample type
            information.
        batch_column : str or None
            The name of the column in the sample metadata with the batch
            information.
        random_state : int, RandomState instance, or None
            The random seed used for all methods with a random component.
        n_correlated_metabolites : int
            The number of metabolites with the highest correlation to the
            metabolite to be normalized.
        _metabolites : list of str
            List with the names of the metabolites.
        _dataset : pandas DataFrame
            DataFrame with the metabolomic data and the sample metadata.
        _metabolite_dict : dict
            Dictionary with the mapping between the original column names and
            the new column names (MET_1, MET_2, etc.).
        _sample_metadata : pandas DataFrame containing the sample metadata.
        corrs_qc : pandas DataFrame
            DataFrame with the Pearson correlation coefficients between the
            metabolites and the batch information.
        corrs_target : pandas DataFrame
            DataFrame with the Pearson correlation coefficients between the
            metabolites and the samples.
        normalized_data : pandas DataFrame
            DataFrame with the normalized data.
        normalized_dataset : pandas DataFrame
            DataFrame with the normalized data and the sample metadata.
        """
        # attributes for the preprocessing
        self.sample_metadata_columns = list(sample_metadata_columns)
        self.sample_type_column = sample_type_column
        self.batch_column = batch_column
        # attributes for the analysis
        self.random_state = random_state
        self.threads = threads
        self.n_correlated_metabolites = n_correlated_metabolites
        # internal class attributes obtained from preprocessing
        self._metabolites = None
        self._dataset = None
        self._metabolite_dict = None
        self._sample_metadata = None
        self._corrs_qc = None
        self._corrs_target = None
        self.normalized_data = None
        self.normalized_dataset = None
        self.n_features_ = None

    def fit(self, X):
        """
        Fit the transformer on the data X and returns self.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input dataset.

        Returns
        -------
        self

        """

        return self._fit(X)

    def transform(self, X, return_data_only=False):
        """
        Apply the SERRF normalization to the data X.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_transformed : pandas DataFrame of shape (n_samples, n_features)
            The transformed data.

        """
        check_is_fitted(self, "n_features_")

        return self._transform(X, return_data_only=return_data_only)

    def fit_transform(self, X, return_data_only=False):
        """
        Fit the transformer on the data X and returns the transformed data.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input dataset.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        return_data_only : bool, default=False
            If True, the transformed data is returned, else the transformed
            dataset with the sample metadata is returned.

        Returns
        -------
        X_transformed : pandas DataFrame of shape (n_samples, n_features)
            The transformed data.

        """
        self._fit(X)
        return self._transform(X, return_data_only=return_data_only)

    def _fit(self, dataset):
        if self.batch_column is None:
            if "batch" not in self.sample_metadata_columns:
                # Add a batch column if it is not already present
                dataset["batch"] = 1
                self.sample_metadata_columns.append("batch")
            else:
                self.batch_column = "batch"
        dataset[self.sample_type_column] = dataset[self.sample_type_column].str.lower()
        self._sample_metadata = dataset[self.sample_metadata_columns]
        data = dataset.drop(columns=self.sample_metadata_columns)
        data = data.astype(float)

        # Create new column names with prefix 'MET_'
        newcolumns = [f"MET_{i}" for i in range(1, len(data.columns) + 1)]

        # Store the mapping between the new and old column names
        self._metabolite_dict = dict(zip(newcolumns, data.columns))

        # Rename the columns
        data.columns = newcolumns

        # Store the list of metabolite names
        self._metabolites = list(data.columns)
        self.n_features_ = len(self._metabolites)
        # Concatenate the metadata and data to form the preprocessed dataset
        self._dataset = pd.concat([self._sample_metadata, data], axis=1)

        self._corrs_qc, self._corrs_target = self._get_corrs_by_sample_type_and_batch()

    def _transform(self, X, return_data_only):
        """
        Apply the transformation on the data X.

        Parameters
        ----------
        X : pandas DataFrame of shape (n_samples, n_features)
            The input dataset.
        return_data_only : bool, default=False
            If True, the transformed data is returned, else the transformed
            dataset with the sample metadata is returned.

        Returns
        -------
        X_transformed : pandas DataFrame of shape (n_samples, n_features) or
            pandas DataFrame of shape (n_samples, n_features + n_sample_metadata_columns)
            The transformed data or the transformed dataset with the sample
            metadata.

        """

        with Pool(
            processes=self.threads,
        ) as p:
            normalized_metabolites = list(
                tqdm(
                    p.imap(self._normalize_metabolite_parallel, self._metabolites),
                    total=len(self._metabolites),
                )
            )
        self.normalized_data = pd.concat(normalized_metabolites, axis=1)
        self.normalized_data.columns = [
            self._metabolite_dict[i] for i in self.normalized_data.columns
        ]
        self.normalized_dataset = pd.concat(
            [self._sample_metadata, self.normalized_data], axis=1
        )
        if return_data_only:
            X = self.normalized_data
        else:
            X = self.normalized_dataset
        return X

    def _center_data(self, data: np.ndarray) -> np.ndarray:
        mean = np.nanmean(data)
        centered = data - mean
        return centered

    def _standard_scaler(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def _group_by_batch(self) -> pd.core.groupby.groupby.GroupBy:
        """
        Group the dataset by batch.

        Returns
        -------
        pd.core.groupby.groupby.GroupBy
            The grouped data.
        """
        return self._dataset.groupby(by="batch")

    def _split_by_sample_type(
        self, group: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Split the given group by sample type.

        Parameters
        ----------
        group : pd.DataFrame
            The data for a single batch.

        Returns
        -------
        pd.DataFrame
            The data for the qc samples in the given batch.
        pd.DataFrame
            The data for the target samples in the given batch.

        """
        qc = group[group["sampleType"] == "qc"]
        target = group[group["sampleType"] != "qc"]
        return qc, target

    def _get_corrs_by_sample_type_and_batch(self):
        """
        Get the correlations by sample type and batch for the given self._dataset data.

        This function calculates the Pearson's r correlation coefficient
        for the qc and target data for each batch.

        Returns
        -------
        corrs_qc : dict
            The correlations for the qc data, keyed by batch.
        corrs_target : dict
            The correlations for the target data, keyed by batch.
        """
        corrs_qc = {}
        corrs_target = {}

        for batch, group in self._group_by_batch():
            # Get the qc and target data for this batch
            qc_group, target_group = self._split_by_sample_type(group)
            qc_group = group[group["sampleType"] == "qc"]
            qc_group_scaled = self._standard_scaler(qc_group[self._metabolites])

            # Calculate the correlations for this batch
            corrs_qc[batch] = qc_group_scaled.corr(method="spearman")
            if len(target_group) == 0:
                # If there is no target data for this batch, set the
                # correlation to NaN
                corrs_target[batch] = np.nan
            else:
                target_group_scaled = self._standard_scaler(
                    target_group[self._metabolites]
                )
                corrs_target[batch] = target_group_scaled.corr(method="spearman")
        return corrs_qc, corrs_target

    def _get_top_metabolites_in_both_correlations(self, series1, series2, num):
        """
        Get the top metabolites that are in both of the given correlations.

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

        def sorted_intersection(list1, list2):
            """
            Return the intersection of two lists sorted by their appearance in
            list1.
            """
            return [i for i in list1 if i in list2]

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

    def _detect_outliers(self, data, threshold=3):
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

    def _get_sorted_correlation(self, correlation_matrix, metabolite):
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

    def _scale_data(self, group, top_correlated):
        """
        Scales the data for a single batch for a single metabolite.

        This function selects the data for the metabolites with
        the highest correlation to the given metabolite. It then
        scales the data using the standard_scaler function.

        Parameters
        ----------
        group : pandas DataFrame
            The data for a single batch.
        top_correlated : iterable
            The names of the metabolites with the highest correlation to the
            given metabolite. These metabolites will be used for the scaling
            process.

        Returns
        -------
        pandas DataFrame
            The scaled data for the given metabolite in the given batch.
        """
        data = group[list(top_correlated)]
        if data.empty:
            return pd.DataFrame()
        return self._standard_scaler(data)

    def _predict_values(self, qc_data_x, qc_data_y, target_data_x):
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
            n_estimators=500, min_samples_leaf=5, random_state=self.random_state
        )
        model.fit(X=qc_data_x, y=qc_data_y, sample_weight=None)
        qc_prediction = model.predict(qc_data_x)
        target_prediction = model.predict(target_data_x)
        return qc_prediction, target_prediction

    def _normalize_qc_and_target(
        self,
        metabolite,
        qc_group,
        qc_prediction,
        target_group,
        target_prediction,
    ):
        """
        Normalizes the qc and target data using the same formula.

        The normalization is based on the mean of the qc data and the median of the
        non-qc data. The normalization is done separately on the qc and target
        data.

        Parameters
        ----------
        metabolite : str
            The name of the metabolite to normalize.
        qc_group : pandas DataFrame
            The qc data group.
        qc_prediction : pandas Series
            The predicted values for the qc data.
        target_group : pandas DataFrame
            The target data group.
        target_prediction : pandas Series
            The predicted values for the target data.

        Returns
        -------
        norm_qc : pandas Series
            The normalized qc data.
        norm_target : pandas Series
            The normalized target data.
        """
        norm_qc = qc_group[metabolite] / (
            (qc_prediction + qc_group[metabolite].mean())
            / self._dataset[self._dataset["sampleType"] == "qc"][metabolite].mean()
        )
        norm_target = target_group[metabolite] / (
            (
                target_prediction
                + target_group[metabolite].mean()
                - target_prediction.mean()
            )
            / self._dataset[self._dataset["sampleType"] != "qc"][metabolite].median()
        )

        # Set negative values to the original value
        norm_target[norm_target < 0] = target_group[metabolite].loc[
            norm_target[norm_target < 0].index
        ]

        # Normalize the median of the qc and target data to the median of the qc data
        norm_qc = norm_qc / (
            norm_qc.median()
            / self._dataset[self._dataset["sampleType"] == "qc"][metabolite].median()
        )
        norm_target = norm_target / (
            norm_target.median()
            / self._dataset[self._dataset["sampleType"] != "qc"][metabolite].median()
        )
        # MANCA FUNZIONE PER RIMPIAZZARE INF O NAN - ORIGINALE QUA SOTTO:
        # norm[!is.finite(norm)] = rnorm(length(norm[!is.finite(norm)]), sd = sd(norm[is.finite(norm)], na.rm = TRUE) * 0.01)

        return norm_qc, norm_target

    def _merge_and_normalize(
        self, metabolite, norm_qc, norm_target, target_group, target_prediction
    ):
        """
        Merges the qc and target data and normalizes the data using the same
        formula as normalize_qc_and_target.

        The normalization is done separately on the qc and target data.

        Parameters
        ----------
        metabolite : str
            The name of the metabolite to normalize.
        norm_qc : pandas Series
            The normalized qc data.
        norm_target : pandas Series
            The normalized target data.
        target_group : pandas DataFrame
            The target data group.
        target_prediction : pandas Series
            The predicted values for the target data.

        Returns
        -------
        norm : pandas Series
            The normalized data.
        """
        norm = pd.concat([norm_qc, norm_target]).sort_index()

        outliers = self._detect_outliers(data=norm, threshold=3)
        outliers = outliers[outliers]
        outliers_in_target = outliers.index.intersection(norm_target.index)

        # Replace outlier values in the target data with the mean of the outlier
        # values in the target data minus the mean of the predicted values for the
        # target data
        replaced_outliers = (
            target_group[metabolite]
            - (
                (target_prediction + target_group[metabolite].mean())
                - (
                    self._dataset[self._dataset["sampleType"] != "qc"][
                        metabolite
                    ].median()
                )
            )
        ).loc[outliers_in_target]

        norm_target.loc[outliers_in_target] = replaced_outliers

        # Set negative values to the original value
        if len(norm_target[norm_target < 0]) > 0:
            norm_target[norm_target < 0] = target_group.loc[
                norm_target[norm_target < 0].index
            ][metabolite]

        norm = pd.concat([norm_qc, norm_target]).sort_index()
        return norm

    def _normalize_metabolite_parallel(self, metabolite):
        return self._normalize_metabolite(metabolite)

    def _normalize_metabolite(self, metabolite):
        """
        Normalizes the given metabolite
        Parameters
        ----------
        metabolite : str
            The name of the metabolite to normalize.

        Returns
        -------
        normalized : pandas Series
            The normalized data.
        """
        normalized_metabolite = []
        for batch, group in self._group_by_batch():
            # Get the groups for the qc and target data
            qc_group, target_group = self._split_by_sample_type(group)

            # Get the order of correlation for the qc and target data
            corr_qc_order = self._get_sorted_correlation(
                self._corrs_qc[batch], metabolite
            )
            corr_target_order = self._get_sorted_correlation(
                self._corrs_target[batch], metabolite
            )

            # Get the top correlated metabolites from both data sets
            top_correlated = self._get_top_metabolites_in_both_correlations(
                corr_target_order, corr_qc_order, 10
            )
            # this is to avoid different order of same 10 metabolites
            # which in turn gives different RF results

            top_correlated = list(top_correlated)
            top_correlated.sort()
            # Scale the data
            target_data_x = self._scale_data(target_group, top_correlated)
            qc_data_x = self._scale_data(qc_group, top_correlated)
            # Get the target values for the qc data
            # qc_data_y = self._get_qc_data_y(qc_group, target_group, metabolite) #outdated function
            qc_data_y = self._center_data(qc_group[metabolite])

            # If there is no QC data, just return the original data
            if qc_data_x.empty:
                norm = group[metabolite]

            # Otherwise, predict and normalize the data
            else:
                qc_prediction, target_prediction = self._predict_values(
                    qc_data_x, qc_data_y, target_data_x
                )
                norm_qc, norm_target = self._normalize_qc_and_target(
                    metabolite,
                    qc_group,
                    qc_prediction,
                    target_group,
                    target_prediction,
                )
                norm = self._merge_and_normalize(
                    metabolite, norm_qc, norm_target, target_group, target_prediction
                )

            normalized_metabolite.append(norm)

        normalized_metabolite = pd.concat(normalized_metabolite)
        return normalized_metabolite
