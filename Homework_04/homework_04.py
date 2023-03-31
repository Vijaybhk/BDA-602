from __future__ import annotations

import sys

import pandas as pd
from dataset_loader import TestDatasets
from pandas import DataFrame
from plotter import VariablePlotter, combine_html
from variables import DfVariableProcessor


def make_clickable(url: str) -> str:
    """
    Function to convert url to hyperlink with name as link
    When you open the link, it opens in a new window as target="_blank"

    Parameters
    -----------
        url : str
            Input url to be converted to hyperlink

    Returns
    -----------
    str
        Clickable hyperlink in html format

    """
    # Source for make clickable and style format:
    # https://stackoverflow.com/questions/42263946/
    # how-to-create-a-table-with-clickable-hyperlink-in-pandas-jupyter-notebook

    return f'<a target="_blank" href="{url}">link</a>'


def predictor_reports(df: DataFrame, predictors: list[str], response: str):
    """
    Function to generate individual reports categorical and continuous predictors
    saves in plots directory

    Parameters
    -----------
        df : DataFrame
            Input dataframe

        predictors: list[str]
            Input list of predictors

        response: str
            Input response variable name

    Returns
    -----------
        None
    """

    # Initializing the Variable processor
    p1 = DfVariableProcessor(input_df=df, predictors=predictors, response=response)

    # Gets the lists of Categorical, Continuous Predictors and Predictor Type Dictionary
    cat_pred, cont_pred, pred_type_dict = p1.get_cat_and_cont_predictors()
    print("Categorical Predictors: ", cat_pred)
    print("Continuous Predictors: ", cont_pred)

    # Gets the response type
    res_type = p1.get_response_type()
    print("Response Type: ", res_type)
    print("Response Variable: ", response)

    # Random Forest Scores, and Regression scores (p values and t scores)
    rf_scores = p1.get_random_forest_scores(cont_pred)
    t_scores, p_values = p1.get_regression_scores(cont_pred)

    # Creating the Final Dataframes
    cont_df = pd.DataFrame(cont_pred, columns=["Features"])
    cont_df["Random Forest Scores"] = cont_df["Features"].map(rf_scores)
    cont_df["p_values"] = cont_df["Features"].map(p_values)
    cont_df["t_scores"] = cont_df["Features"].map(t_scores)

    cat_df = pd.DataFrame(cat_pred, columns=["Features"])

    # Initializing the Plotter with input dataframe
    plot = VariablePlotter(input_df=df)
    plot_dir = plot.create_plot_dir()

    diff_dict, plot_dict, uw_dict, w_dict = plot.get_all_plots(
        cont_pred=cont_pred,
        cat_pred=cat_pred,
        response=response,
        res_type=res_type,
        write_dir=plot_dir,
    )

    name = "categorical"
    for output_df in cat_df, cont_df:
        # Adding all the other columns to the final dataframe
        output_df["Plots"] = output_df["Features"].map(plot_dict)
        output_df["Mean of Response Plot"] = output_df["Features"].map(diff_dict)
        output_df["Diff Mean Response(Weighted)"] = output_df["Features"].map(w_dict)
        output_df["Diff Mean Response(Unweighted)"] = output_df["Features"].map(uw_dict)

        # Ordered Dataframe by Diff Mean Response(Weighted) in descending order
        output_df.sort_values(
            by=["Diff Mean Response(Weighted)"],
            na_position="last",
            ascending=False,
            inplace=True,
            ignore_index=True,
        )

        # Styling the dataframe in html
        # applying the clickable function to the required columns and styling the table.
        output_df.to_html(
            f"{plot_dir}/report_{name}.html",
            formatters={
                "Mean of Response Plot": make_clickable,
                "Plots": make_clickable,
            },
            escape=False,
            index=False,
        )

        name = "continuous"

    return


def main():

    # Change the dataset name below to get the results of dataset you wish from test datasets
    # This will be used as the header in the final html generated.
    data_name = "mpg"

    df, predictors, response = TestDatasets().get_test_data_set(data_set_name=data_name)

    predictor_reports(df=df, predictors=predictors, response=response)

    combine_html(
        combines={
            "Continuous Predictors": "./Plots/report_continuous.html",
            "Categorical Predictors": "./Plots/report_categorical.html",
        },
        result="report.html",
        head=data_name,
    )

    return


if __name__ == "__main__":
    sys.exit(main())
