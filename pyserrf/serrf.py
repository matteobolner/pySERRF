from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np


class SERRF(BaseEstimator, TransformerMixin):
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
      value is 'SampleType'.
    - `batch_column` is the name of the column in the sample metadata with the
      batch information. If `None`, all samples are considered as part the same
      batch. The default value is `None`.
    - `sample_metadata_columns` is a list with the names of the columns in the
      sample metadata; it is important to specify all the metadata columns to
      separate them from the metabolite abundance values.
      The default value is ['SampleType', 'batch', 'label', 'time'].
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
    normalized_data_ : pandas DataFrame
        DataFrame with the normalized data.
    normalized_dataset_ : pandas DataFrame
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
        sample_type_column="SampleType",
        batch_column=None,
        sample_metadata_columns=["SampleType", "batch", "label", "time"],
        random_state=None,
        n_correlated_metabolites=10,
    ):
        """
        Initialize the class.

        Parameters
        ----------
        sample_type_column : str, optional
            The name of the column in the sample metadata with the sample type
            information (i.e qc or normal sample). The default value is
            'SampleType'.
        batch_column : str or None, optional
            The name of the column in the sample metadata with the batch
            information. If None, all samples are considered as part the same
            batch. The default value is None.
        sample_metadata_columns : list of str, optional
            A list with the names of the columns in the sample metadata; it is
            important to specify all the metadata columns to separate them from
            the metabolite abundance values. The default value is
            ['SampleType', 'batch', 'label', 'time'].
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
        normalized_data_ : pandas DataFrame
            DataFrame with the normalized data.
        normalized_dataset_ : pandas DataFrame
            DataFrame with the normalized data and the sample metadata.
        """
        # attributes for the preprocessing
        self.sample_metadata_columns = sample_metadata_columns
        self.sample_type_column = sample_type_column
        self.batch_column = batch_column
        # attributes for the analysis
        self.random_state = random_state
        self.n_correlated_metabolites = n_correlated_metabolites
        # internal class attributes obtained from preprocessing
        self._metabolites = None
        self._dataset = None
        self._metabolite_dict = None
        self._sample_metadata = None
        self._corrs_qc = None
        self._corrs_target = None
        self.normalized_data_ = None
        self.normalized_dataset_ = None

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
            # Add a batch column if it is not already present
            dataset["batch"] = 1
            self.sample_metadata_columns.append("batch")

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

        self._corrs_qc, self._corrs_target = self._get_corrs_by_sample_type_and_batch(
            self._dataset, self._metabolites
        )

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
        counter = 0
        normalized = []
        for metabolite in self._metabolites:
            counter += 1
            print(f"{counter}/{len(self._metabolites)}")
            normalized.append(
                self._normalize_metabolite(
                    self._dataset,
                    metabolite,
                )
            )
        self.normalized_data_ = pd.concat(normalized, axis=1)
        self.normalized_dataset_ = pd.concat(
            [self._sample_metadata, self.normalized_data_], axis=1
        )
        if return_data_only:
            X = self.normalized_data_
        else:
            X = self.normalized_dataset_
        return X

    def _replace_zero_values(
        self,
        values: np.array,
    ) -> pd.Series:
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

        rng = np.random.default_rng(seed=self.random_state)
        np.random.seed(self.random_state)
        # zero_values = row[row == 0].index  # Indices of zero values
        zero_values = np.where(values == 0)[0]
        if len(zero_values) == 0:
            return values
        min_val = np.nanmin(values)  # Minimum non-NaN value in the row=
        mean = min_val + 1  # Mean of the normal distribution
        std = 0.1 * (min_val + 0.1)  # Standard deviation of the normal distribution
        zero_replacements = rng.normal(
            loc=mean,
            scale=std,
            size=len(zero_values),
        )
        values[zero_values] = zero_replacements
        return values

    def _replace_nan_values(self, values) -> pd.Series:
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
        rng = np.random.default_rng(seed=self.random_state)
        np.random.seed(self.random_state)
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

    def _check_for_nan_or_zero(self, data):
        if (0 not in data.values) & (~data.isna().any().any()):
            return False
        return True

    def _handle_zero_and_nan(self, merged, metabolites):
        if self._check_for_nan_or_zero(merged[metabolites]):
            groups = []
            for _, group in merged.groupby(by="batch"):
                if self._check_for_nan_or_zero(group[metabolites]):
                    group[metabolites] = group[metabolites].apply(
                        self._replace_zero_values
                    )
                    group[metabolites] = group[metabolites].apply(
                        self._replace_nan_values
                    )
                    groups.append(group)
                else:
                    groups.append(group)
            merged = pd.concat(groups)
            # CHECK ORDER
        return merged

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

    def _get_corrs_by_sample_type_and_batch(self, merged, metabolites):
        corrs_qc = {}
        corrs_target = {}

        for batch, group in merged.groupby(by="batch"):
            batch_qc = group[group["sampleType"] == "qc"]
            batch_qc_scaled = self._standard_scaler(batch_qc[metabolites])

            corrs_qc[batch] = batch_qc_scaled.corr(method="spearman")
            batch_target = group[group["sampleType"] != "qc"]
            if len(batch_target) == 0:
                corrs_target[batch] = np.nan
            else:
                batch_target_scaled = self._standard_scaler(batch_target[metabolites])
                corrs_target[batch] = batch_target_scaled.corr(method="spearman")
        return corrs_qc, corrs_target

    def _get_top_metabolites_in_both_correlations(self, series1, series2, num):
        selected = set()
        l = num
        while len(selected) < num:
            selected = selected.union(
                set(series1[0:l].index).intersection(set(series2[0:l].index))
            )
            l += 1
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

    def _get_factor(self, qc_group, test_group, metabolite):
        """
        Calculates the factor used to normalize the test data for a single metabolite.

        The factor is calculated as the ratio of the standard deviation of the
        qc data to the standard deviation of the test data.

        Parameters
        ----------
        qc_group : pandas DataFrame
            The qc data for a single batch.
        test_group : pandas DataFrame
            The test data for a single batch.
        metabolite : str
            The name of the metabolite for which the normalization factor is desired.

        Returns
        -------
        float
            The normalization factor for the given metabolite.
        """
        factor = np.std(qc_group[metabolite], ddof=1) / np.std(
            test_group[metabolite], ddof=1
        )
        return factor

    def _get_qc_data_y(self, qc_group, test_group, metabolite):
        # THIS FUNCTION SHOULD BE OUTDATED and UNUSED, IT WAS IN THE UGLY VERSION OF THE CODE
        """
        Calculates the qc data y values for the SERRF regression.

        The qc data y values are calculated as the difference between
        the qc group mean and the actual values, divided by the factor.
        If the factor is smaller than 1 or zero, the center_data function is used
        instead.

        Parameters
        ----------
        qc_group : pandas DataFrame
            The qc data for a single batch.
        test_group : pandas DataFrame
            The test data for a single batch.
        metabolite : str
            The name of the metabolite for which the qc data y values
            are desired.

        Returns
        -------
        pandas Series
            The qc data y values for the given metabolite in the given batch.
        """
        factor = self._get_factor(qc_group, test_group, metabolite)
        # If the factor is smaller than 1 or zero, use center_data function instead
        if (factor == 0) | (factor == np.nan) | (factor < 1):
            qc_data_y = self._center_data(qc_group[metabolite])
        else:
            # If there are more qc data points than in test data,
            # use the qc data mean as a proxy for the test data mean
            if len(qc_group[metabolite]) * 2 > len(test_group[metabolite]):
                qc_data_y = (
                    qc_group[metabolite] - qc_group[metabolite].mean()
                ) / factor
            else:
                qc_data_y = self._center_data(qc_group[metabolite])
        return qc_data_y

    def _predict_values(self, qc_data_x, qc_data_y, test_data_x):
        """
        Predicts the values of the test data using a random forest regressor.

        The qc data is used to build a random forest regressor, which is then
        used to predict the values of the test data. The predicted values are returned
        as a tuple with the qc prediction and the test prediction.

        Parameters
        ----------
        qc_data_x : pandas DataFrame
            The qc data features.
        qc_data_y : pandas Series
            The qc data targets.
        test_data_x : pandas DataFrame
            The test data features.

        Returns
        -------
        tuple
            A tuple containing the qc prediction and the test prediction.
        """
        model = RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, random_state=self.random_state
        )
        model.fit(X=qc_data_x, y=qc_data_y, sample_weight=None)
        qc_prediction = model.predict(qc_data_x)
        test_prediction = model.predict(test_data_x)
        return qc_prediction, test_prediction

    def _normalize_qc_and_test(
        self,
        metabolite,
        merged,
        qc_group,
        qc_prediction,
        test_group,
        test_prediction,
    ):
        """
        Normalizes the qc and test data using the same formula.

        The normalization is based on the mean of the qc data and the median of the
        non-qc data. The normalization is done separately on the qc and test
        data.

        Parameters
        ----------
        metabolite : str
            The name of the metabolite to normalize.
        merged : pandas DataFrame
            The merged data frame containing the data to normalize.
        qc_group : pandas DataFrame
            The qc data group.
        qc_prediction : pandas Series
            The predicted values for the qc data.
        test_group : pandas DataFrame
            The test data group.
        test_prediction : pandas Series
            The predicted values for the test data.

        Returns
        -------
        norm_qc : pandas Series
            The normalized qc data.
        norm_test : pandas Series
            The normalized test data.
        """
        norm_qc = qc_group[metabolite] / (
            (qc_prediction + qc_group[metabolite].mean())
            / merged[merged["sampleType"] == "qc"][metabolite].mean()
        )
        norm_test = test_group[metabolite] / (
            (test_prediction + test_group[metabolite].mean() - test_prediction.mean())
            / merged[merged["sampleType"] != "qc"][metabolite].median()
        )

        # Set negative values to the original value
        norm_test[norm_test < 0] = test_group[metabolite].loc[
            norm_test[norm_test < 0].index
        ]

        # Normalize the median of the qc and test data to the median of the qc data
        norm_qc = norm_qc / (
            norm_qc.median() / merged[merged["sampleType"] == "qc"][metabolite].median()
        )
        norm_test = norm_test / (
            norm_test.median()
            / merged[merged["sampleType"] != "qc"][metabolite].median()
        )
        # MANCA FUNZIONE PER RIMPIAZZARE INF O NAN - ORIGINALE QUA SOTTO:
        # norm[!is.finite(norm)] = rnorm(length(norm[!is.finite(norm)]), sd = sd(norm[is.finite(norm)], na.rm = TRUE) * 0.01)

        return norm_qc, norm_test

    def _merge_and_normalize(
        self, metabolite, norm_qc, norm_test, test_group, test_prediction
    ):
        """
        Merges the qc and test data and normalizes the data using the same
        formula as normalize_qc_and_test.

        The normalization is done separately on the qc and test data.

        Parameters
        ----------
        metabolite : str
            The name of the metabolite to normalize.
        norm_qc : pandas Series
            The normalized qc data.
        norm_test : pandas Series
            The normalized test data.
        test_group : pandas DataFrame
            The test data group.
        test_prediction : pandas Series
            The predicted values for the test data.

        Returns
        -------
        norm : pandas Series
            The normalized data.
        """
        norm = pd.concat([norm_qc, norm_test]).sort_index()

        outliers = self._detect_outliers(data=norm, threshold=3)
        outliers = outliers[outliers]
        outliers_in_test = outliers.index.intersection(norm_test.index)

        # Replace outlier values in the test data with the mean of the outlier
        # values in the test data minus the mean of the predicted values for the
        # test data
        replaced_outliers = (
            test_group[metabolite]
            - (
                (test_prediction + test_group[metabolite].mean())
                - (
                    self._dataset[self._dataset["sampleType"] != "qc"][
                        metabolite
                    ].median()
                )
            )
        ).loc[outliers_in_test]

        norm_test.loc[outliers.index] = replaced_outliers

        # Set negative values to the original value
        if len(norm_test[norm_test < 0]) > 0:
            norm_test[norm_test < 0] = test_group.loc[norm_test[norm_test < 0].index][
                metabolite
            ]

        norm = pd.concat([norm_qc, norm_test]).sort_index()
        return norm

    def _adjust_normalized_values(self, metabolite, merged, normalized):
        ###PROBABLY TO REMOVE
        """
        Adjusts the normalized values for the given metabolite using the median of
        the normalized values for the non-qc data and the qc data.

        The adjustment is done by multiplying the normalized values for the qc
        data by a factor, which is calculated from the standard deviation of the
        normalized values for the non-qc data and the median of the normalized
        values for the qc data.

        The factor is calculated as follows:
            c = (
                normalized_median_non_qc_data
                + (
                    qc_data_median - non_qc_data_median
                ) / non_qc_data_stddev
                * normalized_stddev_non_qc_data
            ) / qc_data_median

        If c > 0, the normalized values for the qc data are multiplied by c.

        Parameters
        ----------
        metabolite : str
            The name of the metabolite to adjust.
        merged : pandas DataFrame
            The merged data frame containing the data to adjust.
        normalized : pandas Series
            The normalized data.

        Returns
        -------
        normalized : pandas Series
            The adjusted normalized data.
        """
        c = (
            normalized.loc[merged[merged["sampleType"] != "qc"].index].median()
            + (
                merged[merged["sampleType"] == "qc"][metabolite].median()
                - merged[merged["sampleType"] != "qc"][metabolite].median()
            )
            / merged[merged["sampleType"] != "qc"][metabolite].std(ddof=1)
            * normalized.loc[merged[merged["sampleType"] != "qc"].index].std()
        ) / normalized.loc[merged[merged["sampleType"] == "qc"].index].median()
        if c > 0:
            normalized.loc[merged[merged["sampleType"] == "qc"].index] = (
                normalized.loc[merged[merged["sampleType"] == "qc"].index] * c
            )
        return normalized

    def _normalize_metabolite(self, merged, metabolite):
        """
        Normalizes the given metabolite by predicting its values using the qc data and
        the other metabolites that are correlated with it.

        Parameters
        ----------
        merged : pandas DataFrame
            The merged data frame containing the data to normalize.
        metabolite : str
            The name of the metabolite to normalize.

        Returns
        -------
        normalized : pandas Series
            The normalized data.
        """
        normalized = []
        for batch, group in merged.groupby(by="batch"):
            # Get the groups for the qc and test data
            qc_group = group[group["sampleType"] == "qc"]
            test_group = group[group["sampleType"] != "qc"]

            # Get the order of correlation for the qc and target data
            corr_qc_order = self._get_sorted_correlation(
                self._corrs_qc[batch], metabolite
            )
            corr_target_order = self._get_sorted_correlation(
                self._corrs_target[batch], metabolite
            )

            # Get the top correlated metabolites from both data sets
            top_correlated = self._get_top_metabolites_in_both_correlations(
                corr_qc_order, corr_target_order, 10
            )
            # this is to avoid different order of same 10 metabolites
            # which in turn gives different RF results

            top_correlated = list(top_correlated)
            top_correlated.sort()
            # Scale the data
            test_data_x = self._scale_data(test_group, top_correlated)
            qc_data_x = self._scale_data(qc_group, top_correlated)
            # Get the target values for the qc data
            # qc_data_y = self._get_qc_data_y(qc_group, test_group, metabolite) #outdated function
            qc_data_y = self._center_data(qc_group[metabolite])

            # If there is no QC data, just return the original data
            if qc_data_x.empty:
                norm = group[metabolite]

            # Otherwise, predict and normalize the data
            else:
                qc_prediction, test_prediction = self._predict_values(
                    qc_data_x, qc_data_y, test_data_x
                )
                norm_qc, norm_test = self._normalize_qc_and_test(
                    metabolite,
                    merged,
                    qc_group,
                    qc_prediction,
                    test_group,
                    test_prediction,
                )
                norm = self._merge_and_normalize(
                    metabolite, norm_qc, norm_test, test_group, test_prediction
                )

            normalized.append(norm)

        normalized = pd.concat(normalized)
        # Adjust the normalized values
        # normalized = self._adjust_normalized_values(metabolite, merged, normalized)
        return normalized
