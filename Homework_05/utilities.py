from __future__ import annotations

import pandas as pd
import plotter as plt
import sqlalchemy
import variables as var
from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_clickable(url: str) -> str:
    """
    Function to convert url to hyperlink with name as word before ".html"
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

    name = url.split("__")[-1].split(".")[0]

    return f'<a target="_blank" href="{url}">{name}</a>'


def predictor_reports(
    df: DataFrame, predictors: list[str], response: str, plot_dir: str
) -> tuple[list, list]:
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

        plot_dir: str
            Input Plots directory path to store plots and predictor reports as html

    Returns
    -----------
        cat_pred: list[str]
            List of Categorical Predictor names from the input dataframe

        cont_pred: list[str]
            List of Continuous Predictor names from the input dataframe

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

    return cat_pred, cont_pred


def mariadb_df(query, db_user="root", db_pass="x11docker"):  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}:3306/{db_database}"
    )

    sql_engine = sqlalchemy.create_engine(connect_string)

    df = pd.read_sql_query(query, sql_engine)

    return df


def make_pipeline(classifier, scaler=StandardScaler()):
    """
    Pipeline function for classifiers
    :param classifier: Classifier function
    :param scaler: Standard Scaler as default
    :return: Pipeline object for the classifiers
    """
    return Pipeline([("std_scaler", scaler), ("classifier", classifier)])


def model_fit_test(classifier_dict, train_var, train_target, test_var, test_target):
    """
    Function includes series of steps from fit to printing test results
    :param classifier_dict: dictionary object with all classifiers to run through the pipeline
    :param train_var: Training Variables Data
    :param train_target: Training Target Data
    :param test_var: Test Variables Data
    :param test_target: Test Target Data
    """
    for i, j in classifier_dict.items():
        model = make_pipeline(j)
        model.fit(train_var, train_target)
        test_pred = model.predict(test_var)
        test_report = classification_report(y_true=test_target, y_pred=test_pred)
        print("\n Classification report of {} : \n {} \n".format(i, test_report))
    return
