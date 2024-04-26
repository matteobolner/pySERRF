from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np

# train, target, num, batch, time, sampletype, minus


class SERRF(BaseEstimator, TransformerMixin):
    def __init__(self, random_state, sample_metadata_columns=["batch", "sampleType"]):
        self.random_state = random_state
        self.sample_metadata_columns = sample_metadata_columns

    def _preprocess_data(self, merged):
        sample_metadata = merged[self.sample_metadata_columns]
        data = merged.drop(columns=self.sample_metadata_columns)
        data = data.astype(float)
        newcolumns = [f"MET_{i}" for i in range(1, len(data.columns) + 1)]
        self._metabolite_dict = {i: j for i, j in zip(newcolumns, data.columns)}
        data.columns = newcolumns
        self._metabolites = list(data.columns)
        self._merged = pd.concat([sample_metadata, data], axis=1)

    def _fit(self, merged, minus):
        self._preprocess_data(merged)
        corrs_train, corrs_target = self._get_corrs_by_sample_type_and_batch(
            self._merged, self._metabolites
        )

        normalized = []
        for metabolite in self._metabolites:
            print(metabolite)
            normalized.append(
                self._normalize_metabolite(
                    self._merged,
                    metabolite,
                    minus,
                    corrs_target=corrs_target,
                    corrs_train=corrs_train,
                )
            )
        return normalized

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y):
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
        else:
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
        mean = data.mean()
        centered = data - mean
        return centered

    def _standard_scaler(self, data: np.ndarray) -> np.ndarray:
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
        centered = self._center_data(data)
        std = data.std(ddof=1)
        scaled = centered / std
        return scaled

    def _get_corrs_by_sample_type_and_batch(self, merged, metabolites):
        corrs_train = {}
        corrs_target = {}

        for batch, group in merged.groupby(by="batch"):
            batch_training = group[group["sampleType"] == "qc"]
            batch_training_scaled = batch_training[metabolites].apply(
                self._standard_scaler
            )
            corrs_train[batch] = batch_training_scaled.corr(method="spearman")
            batch_target = group[group["sampleType"] != "qc"]
            if len(batch_target) == 0:
                corrs_target[batch] = np.nan
            else:
                batch_target_scaled = batch_target[metabolites].apply(
                    self._standard_scaler
                )
                corrs_target[batch] = batch_target_scaled.corr(method="spearman")
        return corrs_train, corrs_target

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
        else:
            return data.apply(self._standard_scaler)

    def _get_factor(self, training_group, test_group, metabolite):
        """
        Calculates the factor used to normalize the test data for a single metabolite.

        The factor is calculated as the ratio of the standard deviation of the
        training data to the standard deviation of the test data.

        Parameters
        ----------
        training_group : pandas DataFrame
            The training data for a single batch.
        test_group : pandas DataFrame
            The test data for a single batch.
        metabolite : str
            The name of the metabolite for which the normalization factor is desired.

        Returns
        -------
        float
            The normalization factor for the given metabolite.
        """
        factor = np.std(training_group[metabolite], ddof=1) / np.std(
            test_group[metabolite], ddof=1
        )
        return factor

    def _get_training_data_y(self, training_group, test_group, metabolite):
        """
        Calculates the training data y values for the SERRF regression.

        The training data y values are calculated as the difference between
        the training group mean and the actual values, divided by the factor.
        If the factor is smaller than 1 or zero, the center_data function is used
        instead.

        Parameters
        ----------
        training_group : pandas DataFrame
            The training data for a single batch.
        test_group : pandas DataFrame
            The test data for a single batch.
        metabolite : str
            The name of the metabolite for which the training data y values
            are desired.

        Returns
        -------
        pandas Series
            The training data y values for the given metabolite in the given batch.
        """
        factor = self._get_factor(training_group, test_group, metabolite)
        # If the factor is smaller than 1 or zero, use center_data function instead
        if (factor == 0) | (factor == np.nan) | (factor < 1):
            training_data_y = self._center_data(training_group[metabolite])
        else:
            # If there are more training data points than in test data,
            # use the training data mean as a proxy for the test data mean
            if len(training_group[metabolite]) * 2 > len(test_group[metabolite]):
                training_data_y = (
                    training_group[metabolite] - training_group[metabolite].mean()
                ) / factor
            else:
                training_data_y = self._center_data(training_group[metabolite])
        return training_data_y

    def _predict_values(self, training_data_x, training_data_y, test_data_x):
        """
        Predicts the values of the test data using a random forest regressor.

        The training data is used to build a random forest regressor, which is then
        used to predict the values of the test data. The predicted values are returned
        as a tuple with the training prediction and the test prediction.

        Parameters
        ----------
        training_data_x : pandas DataFrame
            The training data features.
        training_data_y : pandas Series
            The training data targets.
        test_data_x : pandas DataFrame
            The test data features.

        Returns
        -------
        tuple
            A tuple containing the training prediction and the test prediction.
        """
        model = RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, random_state=self.random_state
        )
        model.fit(X=training_data_x, y=training_data_y, sample_weight=None)
        training_prediction = model.predict(training_data_x)
        test_prediction = model.predict(test_data_x)

        return training_prediction, test_prediction

    def _normalize_training_and_test(
        self,
        minus,
        metabolite,
        merged,
        training_group,
        training_prediction,
        test_group,
        test_prediction,
    ):
        """
        Normalizes the training and test data using the same formula.

        The normalization is based on the mean of the QC data and the median of the
        non-QC data. The normalization is done separately on the training and test
        data.

        Parameters
        ----------
        minus : bool
            If True, the normalization is done by subtracting the predicted value,
            otherwise the normalization is done by dividing the predicted value.
        metabolite : str
            The name of the metabolite to normalize.
        merged : pandas DataFrame
            The merged data frame containing the data to normalize.
        training_group : pandas DataFrame
            The training data group.
        training_prediction : pandas Series
            The predicted values for the training data.
        test_group : pandas DataFrame
            The test data group.
        test_prediction : pandas Series
            The predicted values for the test data.

        Returns
        -------
        norm_training : pandas Series
            The normalized training data.
        norm_test : pandas Series
            The normalized test data.
        """
        if minus:
            norm_training = training_group[metabolite] - (
                (training_prediction + training_group[metabolite].mean())
                - merged[merged["sampleType"] == "qc"][metabolite].mean()
            )
            norm_test = test_group[metabolite] - (
                (test_prediction + test_group[metabolite].mean())
                - merged[merged["sampleType"] != "qc"][metabolite].median()
            )
        else:
            norm_training = training_group[metabolite] / (
                (training_prediction + training_group[metabolite].mean())
                / merged[merged["sampleType"] == "qc"][metabolite].mean()
            )
            norm_test = test_group[metabolite] / (
                (
                    test_prediction
                    + test_group[metabolite].mean()
                    - test_prediction.mean()
                )
                / merged[merged["sampleType"] != "qc"][metabolite].median()
            )

        # Set negative values to the original value
        norm_test[norm_test < 0] = test_group[metabolite].loc[
            norm_test[norm_test < 0].index
        ]

        # Normalize the median of the training and test data to the median of the QC data
        norm_training = norm_training / (
            norm_training.median()
            / merged[merged["sampleType"] == "qc"][metabolite].median()
        )
        norm_test = norm_test / (
            norm_test.median()
            / merged[merged["sampleType"] != "qc"][metabolite].median()
        )

        return norm_training, norm_test

    def _merge_and_normalize(
        self, metabolite, norm_training, norm_test, test_group, test_prediction
    ):
        """
        Merges the training and test data and normalizes the data using the same
        formula as normalize_training_and_test.

        The normalization is done separately on the training and test data.

        Parameters
        ----------
        metabolite : str
            The name of the metabolite to normalize.
        norm_training : pandas Series
            The normalized training data.
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
        norm = pd.concat([norm_training, norm_test]).sort_index()

        ##NOT SURE ABOUT LINE BELOW, WORKS AS INTENDED BUT VALUES GENERATED ARE VERY LOW (CENTERED ON ZERO); MIGHT BENEFIT FROM NORM MEAN AS CENTER
        # norm[~np.isfinite(norm)] = rng.normal(
        #    scale=np.std(norm[np.isfinite(norm)], ddof=1) * 0.01,
        #    size=len(norm[~np.isfinite(norm)]),
        # )

        outliers = self._detect_outliers(data=norm, threshold=3)
        outliers = outliers[outliers]
        outliers_in_test = outliers.index.intersection(norm_test.index)

        # Replace outlier values in the test data with the mean of the outlier
        # values in the test data minus the mean of the predicted values for the
        # test data
        attempt = (
            test_group[metabolite]
            - (
                (test_prediction + test_group[metabolite].mean())
                - (
                    self._merged[self._merged["sampleType"] != "qc"][
                        metabolite
                    ].median()
                )
            )
        ).loc[outliers_in_test]

        if len(outliers) > 0 & len(attempt) > 0:
            if outliers.mean() > norm.mean():
                if attempt.mean() < outliers.mean():
                    norm_test.loc[outliers.index] = attempt
            else:
                if attempt.mean() > outliers.mean():
                    norm_test.loc[outliers.index] = attempt

        # Set negative values to the original value
        if len(norm_test[norm_test < 0]) > 0:
            norm_test[norm_test < 0] = test_group.loc[norm_test[norm_test < 0].index][
                metabolite
            ]

        norm = pd.concat([norm_training, norm_test]).sort_index()
        return norm

    def _adjust_normalized_values(self, metabolite, merged, normalized):
        """
        Adjusts the normalized values for the given metabolite using the median of
        the normalized values for the non-QC data and the QC data.

        The adjustment is done by multiplying the normalized values for the QC
        data by a factor, which is calculated from the standard deviation of the
        normalized values for the non-QC data and the median of the normalized
        values for the QC data.

        The factor is calculated as follows:
            c = (
                normalized_median_non_qc_data
                + (
                    qc_data_median - non_qc_data_median
                ) / non_qc_data_stddev
                * normalized_stddev_non_qc_data
            ) / qc_data_median

        If c > 0, the normalized values for the QC data are multiplied by c.

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

    def _normalize_metabolite(
        self, merged, metabolite, minus, corrs_train, corrs_target
    ):
        """
        Normalizes the given metabolite by predicting its values using the training data and
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
            # Get the groups for the training and test data
            training_group = group[group["sampleType"] == "qc"]
            test_group = group[group["sampleType"] != "qc"]

            # Get the order of correlation for the training and target data
            corr_train_order = self._get_sorted_correlation(
                corrs_train[batch], metabolite
            )
            corr_target_order = self._get_sorted_correlation(
                corrs_target[batch], metabolite
            )

            # Get the top correlated metabolites from both data sets
            top_correlated = self._get_top_metabolites_in_both_correlations(
                corr_train_order, corr_target_order, 10
            )

            # Scale the data
            training_data_x = self._scale_data(training_group, top_correlated)
            test_data_x = self._scale_data(test_group, top_correlated)

            # Get the target values for the training data
            training_data_y = self._get_training_data_y(
                training_group, test_group, metabolite
            )

            # If there is no correlation data, just return the original data
            if training_data_x.empty:
                norm = group[metabolite]

            # Otherwise, predict and normalize the data
            else:
                training_prediction, test_prediction = self._predict_values(
                    training_data_x, training_data_y, test_data_x
                )
                norm_training, norm_test = self._normalize_training_and_test(
                    minus,
                    metabolite,
                    merged,
                    training_group,
                    training_prediction,
                    test_group,
                    test_prediction,
                )
                norm = self._merge_and_normalize(
                    metabolite, norm_training, norm_test, test_group, test_prediction
                )

            normalized.append(norm)

        normalized = pd.concat(normalized)

        # Adjust the normalized values
        normalized = self._adjust_normalized_values(metabolite, merged, normalized)
        return normalized
