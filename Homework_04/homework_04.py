import sys

import pandas as pd
from dataset_loader import TestDatasets
from plotter import VariablePlotter
from variables import DfVariableProcessor


# Hyperlink clickable function
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
    return f'<a target="_blank" href="{url}">link</a>'


def main():

    df, predictors, response = TestDatasets().get_test_data_set(data_set_name="mpg")
    # Change the dataset name above to get the results of dataset you wish from test datasets

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

    # Initializing the Plotter with input dataframe
    plot = VariablePlotter(input_df=df)
    plot_dir = plot.create_plot_dir()

    diff_dict, plot_dict = plot.get_all_plots(
        cont_pred=cont_pred,
        cat_pred=cat_pred,
        response=response,
        res_type=res_type,
        write_dir=plot_dir,
    )

    # Creating the Final Dataframe using predictors and their types
    output_df = pd.DataFrame.from_dict(
        pred_type_dict, orient="index", columns=["Predictor Type"]
    )

    output_df.reset_index(names="Variable", inplace=True)

    # Adding all the other columns to the final dataframe
    output_df["Response Variable"] = response
    output_df["Plots"] = output_df["Variable"].map(plot_dict)
    output_df["Random Forest Scores"] = output_df["Variable"].map(rf_scores)
    output_df["p_values"] = output_df["Variable"].map(p_values)
    output_df["t_scores"] = output_df["Variable"].map(t_scores)
    output_df["Diff with Mean of Response"] = output_df["Variable"].map(diff_dict)

    # Ordered Dataframe by Random Forest Scores in descending order
    output_df.sort_values(
        by=["Random Forest Scores"],
        na_position="last",
        ascending=False,
        inplace=True,
        ignore_index=True,
    )

    # Source for make clickable and style format:
    # https://stackoverflow.com/questions/42263946/
    # how-to-create-a-table-with-clickable-hyperlink-in-pandas-jupyter-notebook

    # Styling the dataframe in html
    # applying the clickable function to the required columns and styling the table.
    output_df.to_html(
        "report.html",
        formatters={
            "Diff with Mean of Response": make_clickable,
            "Plots": make_clickable,
        },
        escape=False,
    )

    output_df_styler = output_df.style
    output_df_styler.set_caption(
        ("Predictors Plots and Scores ordered by Random Forest", "bold")
    )

    output_df_styler.format(
        na_rep="na",
        formatter={
            "Diff with Mean of Response": make_clickable,
            "Plots": make_clickable,
        },
    )

    output_df_styler.set_table_styles(
        [
            {"selector": ",th,td", "props": [("border", "1px solid grey")]},
            {
                "selector": "caption",
                "props": [("font-weight", "bold"), ("font-size", "20px")],
            },
        ]
    )

    # Saves styled dataframe html locally
    output_df_styler.to_html("report_with_caption.html")

    return


if __name__ == "__main__":
    sys.exit(main())
