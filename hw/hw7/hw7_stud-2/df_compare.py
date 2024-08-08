import pandas as pd


def standardize_df(df):
    """ organizes a dataframe in standard way, allows for index-less comparison
    
    returns a copy of the dataframe which:
    - sorts columns alphabetically (left to right)
    - sorts rows by increasing values on all columns (top to bottom)
    - resets index
    
    Args:
        df (pd.DataFrame): an input dataframe
    """
    df = df.copy()
    df = df.loc[:, sorted(df.columns)]
    df.sort_values(axis=0, inplace=True, ascending=True, by=list(df.columns))
    df.reset_index(inplace=True, drop=True)

    return df


def assert_df_equal_no_idx(df0, df1):
    """ compares whether two dataframe are equal (ignoring index)
    
    
       
    Args:
        df0 (pd.DataFrame): an input dataframe
        df1 (pd.DataFrame): an input dataframe
    """

    # standardize each df
    df0 = standardize_df(df0)
    df1 = standardize_df(df1)

    pd.testing.assert_frame_equal(df0, df1)


if __name__ == '__main__':
    """ in terms of row content, two dataframes below are equivalent (a row 
    where a=1, b=3 and another row where a=2, b=4).   however, due to unique
    indexing and row ordering, its tough to compare them to identify this row
    equivalence.  The assert_df_equal_no_idx() fnc above first standardize_df()
    each so that indexing differences won't interrupt showing their equivilence
    """
    df0 = pd.DataFrame({'a': [1, 2],
                        'b': [3, 4]},
                       index=[0, 1])

    df1 = pd.DataFrame({'b': [4, 3],
                        'a': [2, 1]},
                       index=['different_index0', 'different_index1'])

    assert_df_equal_no_idx(df0, df1)
