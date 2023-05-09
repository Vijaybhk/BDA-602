from __future__ import annotations

import pandas as pd
import plotter as plt
import variables as var
from correlation_bruteforce import CorrelationAndBruteForce
from pandas import DataFrame
from plotter import combine_html
from utilities import make_clickable


def generate_report(
    df: DataFrame, predictors: list[str], response: str, plot_dir: str, data_name: str
):
    """
    Function to generate individual reports categorical and continuous predictors
    saves in plots directory.
    Combines all brute force and correlation reports to the predictor reports.
    And creates the final html in the root directory as data_name.html

    Parameters
    -----------
        df : DataFrame
            Input dataframe

        predictors: list[str]
            Input list of predictors

        response: str
            Input response variable name

        plot_dir: str
            Input Plots directory path to store plots and predictor reports as html

        data_name: str
            Name of the dataset, uses a header in the html report

    Returns
    -----------
        None

    """

    # Initializing the Variable processor
    p1 = var.DfVariableProcessor(input_df=df, predictors=predictors, response=response)

    # Gets the lists of Categorical, Continuous Predictors and Predictor Type Dictionary
    cat_pred, cont_pred, _ = p1.get_cat_and_cont_predictors()
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
    plot = plt.VariablePlotter(input_df=df)

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
        output_df["Diff Mean Response (Weighted)"] = output_df["Features"].map(w_dict)
        output_df["Diff Mean Response (Unweighted)"] = output_df["Features"].map(
            uw_dict
        )

        # Ordered Dataframe by Diff Mean Response(Weighted) in descending order
        output_df.sort_values(
            by=["Diff Mean Response (Weighted)"],
            na_position="last",
            ascending=False,
            inplace=True,
            ignore_index=True,
        )

        # Styling the dataframe in html
        # applying the clickable function to the required columns.
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

    # Instantiating CorrelationAndBruteForce class with inputs
    corr_bf = CorrelationAndBruteForce(
        input_df=df, predictors=predictors, response=response, write_dir=plot_dir
    )

    # Gets all correlation, brute force tables and matrices as html
    corr_bf.get_all_correlation_metrics()
    corr_bf.get_all_brute_force_metrics()

    # Empty dictionaries
    cat_rep, cat_corr, cat_brut, cont_corr, cont_brut, cont_rep = {}, {}, {}, {}, {}, {}

    # Headings of html as key
    # Relative path of html as value

    # If there is at least one Categorical predictor, gets all cat dictionaries
    if len(cat_pred) > 0:
        cat_corr = {
            "<h2> Categorical/Categorical Correlations </h2>"
            "<h3> Correlation Tschuprow Matrix </h3>": f"{plot_dir}/Cat_Cat_Tschuprow_matrix.html",
            "<h3> Correlation Cramer Matrix </h3>": f"{plot_dir}/Cat_Cat_Cramer_matrix.html",
            "<div class='row'> <div class='column'>"
            "<h3> Correlation Tschuprow Table </h3>": f"{plot_dir}/Cat_Cat_Tschuprow_corr.html",
            "</div> <div class='column'>"
            " <h3> Correlation Cramer Table </h3>": f"{plot_dir}/Cat_Cat_Cramer_corr.html",
            "</div> <h2> Categorical/Continuous Correlations </h2>"
            "<h3> Correlation Ratio Matrix </h3>": f"{plot_dir}/Cat_Cont_matrix.html",
            "<h3> Correlation Ratio Table </h3>": f"{plot_dir}/Cat_Cont_corr.html",
        }

        cat_brut = {
            "<h2> Categorical/Categorical Brute Force </h2>": f"{plot_dir}/Cat_Cat_brute.html",
            "<h2> Categorical/Continuous Brute Force </h2>": f"{plot_dir}/Cat_Cont_brute.html",
        }

        cat_rep = {
            "<h2> Categorical Predictors </h2>": f"{plot_dir}/report_categorical.html"
        }

    # If there is at least one Continuous predictor, gets all cont dictionaries
    if len(cont_pred) > 0:
        cont_corr = {
            "<h2> Continuous/Continuous Correlations </h2>"
            "<h3> Correlation Pearson's Matrix </h3>": f"{plot_dir}/Cont_Cont_matrix.html",
            "<h3> Correlation Pearson's Table </h3>": f"{plot_dir}/Cont_Cont_corr.html",
        }

        cont_brut = {
            "<h2> Continuous/Continuous Brute Force </h2>": f"{plot_dir}/Cont_Cont_brute.html"
        }

        cont_rep = {
            "<h2> Continuous Predictors </h2>": f"{plot_dir}/report_continuous.html"
        }

    # Unpacks all cat and cont dictionaries in order to get the final "report.html"
    # inside the combine html function, and "data name" as the main heading
    combine_html(
        combines={
            **cont_rep,
            **cat_rep,
            **cat_corr,
            **cont_corr,
            **cat_brut,
            **cont_brut,
        },
        result=f"Output/{data_name}.html",
        head=data_name,
    )

    return
