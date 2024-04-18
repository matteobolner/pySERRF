import pandas as pd
import numpy as np

def read_serff_format_data_simple(input_path):
    data = pd.read_excel(input_path, header=None)
    data.replace("", np.nan, inplace=True)
    data=data.transpose()
    data.columns=data.iloc[1]
    data=data.iloc[2::]
    data.columns.name=None
    data=data.reset_index(drop=True)
    return data


def rename_duplicate_columns(df):
    """
    Renames duplicate columns by appending an incrementing integer to the end of the
    column name. For example, if you have two columns named "A", the first occurrence
    will be renamed "A", and the second occurrence will be renamed "A_1".

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to rename duplicate columns in

    Returns
    -------
    pandas.DataFrame
        DataFrame with duplicate columns renamed
    """
    s = df.columns.to_series().groupby(df.columns)
    df.columns = np.where(
        s.transform("size") > 1, df.columns + "_" + s.cumcount().astype(str), df.columns
    )
    df.columns = [i.replace("_0", "") for i in df.columns]
    return df


def read_serff_format_data(path, sheet=0):
    """
    Read data in SERRF format from a file or Excel sheet

    Parameters
    ----------
    path : str
        Path to the input file. If the file is in Excel format, specify the sheet name or index.
    sheet : int, str, optional
        Sheet name or index of the sheet to be read. Default is the first sheet.

    Returns
    -------
    result : dict
        A dictionary with four keys: 'p', 'f', 'e', and 'e_matrix'.
        The value of each key is a Pandas DataFrame.

        * 'p' is the sample information table
        * 'f' is the compound information table
        * 'e' is the response table with samples as rows and compounds as columns
        * 'e_matrix' is the response matrix with samples as rows and compounds as columns

        If there is no 'label' column in the data, raise a ValueError.

        All tables are stripped of whitespaces in the 'label' column.

    Raises
    ------
    ValueError
        If the data does not have 'label' column.
    """
    if path.endswith(".xlsx"):
        data = pd.read_excel(path, header=None, sheet_name=sheet)
    else:
        data = pd.read_csv(path, header=None)

    data.replace("", np.nan, inplace=True)

    sample_col_range = data.columns[data.iloc[0].first_valid_index() : :]
    sample_row_range = data.index[data.iloc[:, 0].first_valid_index() :: -1]

    compound_col_range = data.columns[0 : sample_col_range[0] + 1]
    # compound_col_range=data.iloc[:, :sample_col_range].last_valid_index() + 1
    compound_row_range = data.index[
        data.index[sample_row_range[0]] : data.last_valid_index() + 1
    ]

    #####GET P
    p = data.loc[sample_row_range, sample_col_range].T
    # p = data.iloc[sample_row_range:, sample_col_range:].T
    p.columns = p.iloc[0]

    p = p.iloc[1:]

    # p = p[[p.columns[-1]] + list(p.columns[:-1])]
    p.reset_index(drop=True, inplace=True)
    p.index.name = None
    p.columns.name = None

    p = rename_duplicate_columns(p)

    if "label" not in p.columns:
        raise ValueError(
            "Cannot find 'label' in your data. Please check the data format requirement."
        )

    p["label"].fillna("na", inplace=True)
    p["label"] = p["label"].str.rstrip()

    #####GET F
    f = data.loc[compound_row_range, compound_col_range]
    f.columns = f.iloc[0].astype(str)
    f = f.iloc[1:]
    f = f[[f.columns[-1]] + list(f.columns[:-1])]
    f.reset_index(drop=True, inplace=True)
    f.index.name = None
    f.columns.name = None

    f = rename_duplicate_columns(f)

    f["label"].fillna("na", inplace=True)
    f["label"] = f["label"].str.rstrip()

    #####GET E

    e = data.iloc[compound_row_range, sample_col_range]
    e.columns = e.iloc[0].astype(str).fillna("na")
    e = e.iloc[1:]
    e = pd.concat([e["label"], e.iloc[:, 1:].apply(pd.to_numeric)], axis=1)

    e.reset_index(drop=True, inplace=True)
    e.index.name = None
    e.columns.name = None

    e["label"].fillna("na", inplace=True)
    e["label"] = e["label"].str.rstrip()
    e.columns = ["label"] + list(p["label"])
    e = rename_duplicate_columns(e)
    e_matrix = e.iloc[:, 1:]

    return {"p": p, "f": f, "e": e, "e_matrix": e_matrix}
