import pandas as pd
import numpy as np


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
    # TODO add documentation
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


path = "test_data/SERRF example dataset.xlsx"
sheet = 0

test = read_serff_format_data(path, sheet)

for i in "e", "f", "e_matrix", "p":
    temp_r = pd.read_table(f"test_data/{i}.tsv").fillna("na")
    temp_r = temp_r.reset_index(drop=True)
    temp_r.columns = [i.replace(".", "_") for i in temp_r.columns]
    tempdf = test[i].reset_index(drop=True).fillna("na")

    if not temp_r.equals(tempdf):
        print(f"{i} is different")
        break
