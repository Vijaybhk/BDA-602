import sys

from dataset_loader import TestDatasets
from plotter import VariablePlotter
from variables import DfVariableProcessor


def main():

    df, predictors, response = TestDatasets().get_test_data_set(data_set_name="mpg")
    # Change the dataset name above to get the results of dataset you wish from test datasets

    p1 = DfVariableProcessor(input_df=df, predictors=predictors, response=response)
    cat_pred, cont_pred = p1.get_cat_and_cont_predictors()
    res_type = p1.get_response_type()

    print("Categorical Predictors: ", cat_pred)
    print("Continuous Predictors: ", cont_pred)
    print("Response Type: ", res_type)
    print("Response Variable: ", response)

    plot = VariablePlotter(input_df=df)
    plot_dir = plot.create_plot_dir()

    for i in cont_pred:
        if res_type == "categorical":
            plot.cat_response_cont_predictor(response, i, plot_dir)

        elif res_type == "continuous":
            plot.cont_response_cont_predictor(response, i, plot_dir)

    for j in cat_pred:
        if res_type == "categorical":
            plot.cat_response_cat_predictor(response, j, plot_dir)

        elif res_type == "continuous":
            plot.cont_response_cat_predictor(response, j, plot_dir)

    return


if __name__ == "__main__":
    sys.exit(main())
