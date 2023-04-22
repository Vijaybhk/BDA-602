from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
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
    if len(name) > 20:
        name = "link"

    return f'<a target="_blank" href="{url}">{name}</a>'


def mariadb_df(
    query,
    db_user="root",
    db_pass="x11docker",  # pragma: allowlist secret
    db_host="localhost",
    db_database="baseball",
):

    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )

    sql_engine = create_engine(connect_string)

    # Source: https://stackoverflow.com/questions/75310173/attributeerror-optionengine-object-has-no-attribute-execute
    # df = pd.read_sql_query(query, sql_engine)
    # above line won't run, and needs below lines to fix it

    with sql_engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)

    return df


def model_results(name, classifier, x_train, x_test, y_train, y_test, write_dir):
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
