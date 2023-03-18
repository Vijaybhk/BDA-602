import sys

import pandas as pd
from dataset_loader import TestDatasets
from plotter import VariablePlotter
from variables import DfVariableProcessor


def make_clickable(url):
    return f'<a target="_blank" href="{url}">link</a>'


def main():

    df, predictors, response = TestDatasets().get_test_data_set(data_set_name="mpg")
    # Change the dataset name above to get the results of dataset you wish from test datasets

    p1 = DfVariableProcessor(input_df=df, predictors=predictors, response=response)
    cat_pred, cont_pred = p1.get_cat_and_cont_predictors()
    print("Categorical Predictors: ", cat_pred)
    print("Continuous Predictors: ", cont_pred)

    res_type = p1.get_response_type()
    print("Response Type: ", res_type)
    print("Response Variable: ", response)

    plot = VariablePlotter(input_df=df)
    plot_dir = plot.create_plot_dir()

    rf_scores = p1.get_random_forest_scores(cont_pred, response)
    t_scores, p_values = p1.get_regression_scores(cont_pred)

    diff_dict = {}
    plot_dict = {}

    for i in cont_pred:
        if res_type == "categorical":
            plot.cat_response_cont_predictor(response, i, plot_dir)
            diff_dict[i] = "./Plots/Combined_Diff_of_{}.html".format(i)
            plot_dict[i] = "./Plots/Combined_plot_of_{}.html".format(i)

        elif res_type == "continuous":
            plot.cont_response_cont_predictor(response, i, plot_dir)
            diff_dict[i] = "./Plots/Combined_Diff_of_{}.html".format(i)
            plot_dict[i] = "./Plots/scatter_plot_of_{}.html".format(i)

    for j in cat_pred:
        if res_type == "categorical":
            plot.cat_response_cat_predictor(response, j, plot_dir)
            diff_dict[j] = "./Plots/Diff_Plot_{}_and_{}.html".format(j, response)
            plot_dict[j] = "./Plots/Density_Heat_Map_of_{}.html".format(j)

        elif res_type == "continuous":
            plot.cont_response_cat_predictor(response, j, plot_dir)
            diff_dict[j] = "./Plots/Diff_Plot_{}_and_{}.html".format(j, response)
            plot_dict[j] = "./Plots/Combined_plot_of_{}.html".format(j)

    pred_type_dict = {}

    for i in predictors:
        if p1.check_continuous_var(i):
            pred_type_dict[i] = "Continuous"
        else:
            pred_type_dict[i] = "Categorical"

    output_df = pd.DataFrame.from_dict(
        pred_type_dict, orient="index", columns=["Predictor Type"]
    )
    output_df.reset_index(names="Variable", inplace=True)
    output_df["Response Variable"] = response
    output_df["Plots"] = output_df["Variable"].map(plot_dict)
    output_df["Random Forest Scores"] = output_df["Variable"].map(rf_scores)
    output_df["p_values"] = output_df["Variable"].map(p_values)
    output_df["t_scores"] = output_df["Variable"].map(t_scores)
    output_df["Diff with Mean of Response"] = output_df["Variable"].map(diff_dict)

    # Ordered Dataframe by Random Forest Scores in descending order
    output_df = output_df.sort_values(
        by=["Random Forest Scores"], na_position="last", ascending=False
    )
    output_df.reset_index(drop=True, inplace=True)

    output_df = output_df.style.format(
        {"Diff with Mean of Response": make_clickable, "Plots": make_clickable}
    )

    output_df = output_df.set_properties(
        **{
            "color": "black",
            "border": "2px solid grey",
            "background-color": "white",
        }
    )

    output_df.to_html("scores.html", justify="left", render_links=True)

    return


if __name__ == "__main__":
    sys.exit(main())
