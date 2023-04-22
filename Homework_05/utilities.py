from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text


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

    # if length of the name for hyperlink is more than 20, just show link
    if len(name) > 20:
        name = "link"

    return f'<a target="_blank" href="{url}">{name}</a>'


def mariadb_df(
    query: str,
    db_user: str = "root",
    db_pass: str = "x11docker",  # pragma: allowlist secret
    db_host: str = "localhost",
    db_database: str = "baseball",
) -> DataFrame:
    """
    Parameters
    --------------

        query:  str
            Query to get the data from database
        db_user: str
            Username to access the database, default is root
        db_pass: str
            password to access the database, default is x11docker
        db_host: str
            host to access the database, default is localhost
        db_database: str
            name of the database, default in this project is baseball

    Returns
    -------------
        df: DataFrame
            pandas dataframe created from the database using the data from input query

    """

    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )

    sql_engine = create_engine(connect_string)

    # Source: https://stackoverflow.com/questions/75310173/attributeerror-optionengine-object-has-no-attribute-execute
    # df = pd.read_sql_query(query, sql_engine)
    # above line won't run, and needs below lines to fix it, source link provided.

    with sql_engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)

    return df


def model_results(name, classifier, x_train, x_test, y_train, y_test, write_dir):
    """
    Parameters
    --------------

        name:
            Name of the model to input in Sklearn Pipeline
        classifier:
            Classifier model to input in Sklearn Pipeline
        x_train:
            Training data, array of n_features, n_samples
        x_test:
            Testing data, array of n_features, n_samples
        y_train:
            Training data, array of response, n_samples
        y_test:
            Testing data, array of response, n_samples
        write_dir:
            write directory path for the classification report and
            confusion matrices created from the model

    Returns
    -------------

        model
            sklearn pipeline object after training the model.
        predictions
            predicted values array from the model using x_test data

    """
    # Classifier Model in a pipeline
    model = Pipeline([("std_scaler", StandardScaler()), (name, classifier)])

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    cr = classification_report(y_test, predictions, output_dict=True)
    cm = confusion_matrix(y_test, predictions)

    df_cr = pd.DataFrame(cr).transpose()
    df_cr.to_html(f"{write_dir}/{name}_cr.html")

    layout = {
        "title": "<b> Confusion Matrix </b>",
        "xaxis": {"title": "<b> Predicted value </b>"},
        "yaxis": {"title": "<b> True value </b>"},
        "height": 500,
        "paper_bgcolor": "#fafafa",
    }

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["HTLoses", "HTWins"],
            y=["HTLoses", "HTWins"],
            hoverongaps=False,
            texttemplate="%{z}",
        ),
        layout=layout,
    )
    fig.write_html(f"{write_dir}/{name}_cm.html")

    return model, predictions
