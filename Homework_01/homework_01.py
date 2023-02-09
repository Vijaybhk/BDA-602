import sys

import numpy as np
import pandas as pd


def dataframe_to_array(dataframe, a, b):
    """
    Converting dataframe to numpy array
    :param dataframe: input dataframe
    :param a: start number of the columns
    :param b: end number of the columns
    :return: numpy array with all the rows and columns with indexed columns
    """
    return dataframe.iloc[:, a:b].to_numpy()


def numpy_statistics(array, column_list):
    """
    Calculate Summary Statistics using NumPy
    :param array: Input array for which statistics are to be calculated
    :param column_list: List of column names for which statistics are to be calculated
    :return: an output dataframe with indexes as statistics and columns as column names of the primary dataframe
    """
    mean = np.mean(array, axis=0)
    minimum = np.min(array, axis=0)
    maximum = np.max(array, axis=0)
    quantiles = np.quantile(array, [0.25, 0.5, 0.75], axis=0)
    out_arr = np.vstack((mean, minimum, maximum, quantiles))
    out_df = pd.DataFrame(
        data=out_arr,
        index=["mean", "min", "max", "quartile", "median", "third quartile"],
        columns=column_list,
    )
    return out_df


def main():
    # Using URL for data instead of downloading for better reproducibility
    data_path = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    )

    # Data from source does not have column names included
    col_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]

    # Loading dataset into a pandas dataframe
    df = pd.read_csv(data_path, header=None, names=col_names)

    # Checking head and tail of the dataframe
    print(df.head())
    print(df.tail())

    # Summary Statistics using Pandas dataframe describe function
    print(df.describe())

    # Converting the dataframe df to numpy array
    np_array = dataframe_to_array(df, 0, 4)

    # Summary statistics using numpy
    stat_df = numpy_statistics(np_array, col_names[:4])
    print(stat_df)


if __name__ == "__main__":
    sys.exit(main())
