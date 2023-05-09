from __future__ import annotations

import os
import sys

from generator import generate_report
from plotter import combine_html
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from utilities import mariadb_df, model_results


def main():
    # Name of the data set name to pass in as "heading of the html"
    data_name = "Baseball"

    # Getting data from baseball database and created features table
    query1 = "SELECT * FROM" + " team_pitcher_features"
    query2 = "SELECT * FROM" + " team_batter_features"
    query3 = "SELECT * FROM" + " start_pitcher_features"
    df1 = mariadb_df(query=query1, db_host="vij_mariadb:3306")
    df2 = mariadb_df(query=query2, db_host="vij_mariadb:3306")
    df3 = mariadb_df(query=query3, db_host="vij_mariadb:3306")

    df = df1.merge(df2, on=["game_id", "game_date"])
    df = df.merge(df3, on=["game_id", "game_date"])
    df.sort_values(by=["game_date", "game_id"], inplace=True, ignore_index=True)

    # Getting predictors and response from dataframe
    predictors = df.columns[2:-1]
    response = df.columns[-1]

    # Plots directory path
    this_dir = os.path.dirname(os.path.realpath(__file__))
    plot_dir = f"{this_dir}/Output/Plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Creating Predictors report as html
    generate_report(df, predictors, response, plot_dir, data_name)

    # Dividing first 75percent of games as train data
    # As the data is sorted by game id and date, train data will be past data
    # and test data will be for future games
    train_size = int(len(df) * 0.75)
    x_train = df.iloc[:train_size, 2:-1].values
    x_test = df.iloc[train_size:, 2:-1].values
    y_train = df.iloc[:train_size, -1].values
    y_test = df.iloc[train_size:, -1].values

    # Random Forest Model
    rf_model, rf_pred = model_results(
        name="rfc",
        classifier=RandomForestClassifier(random_state=42),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        write_dir=plot_dir,
    )

    print("\nRandom Forest Results:\n")
    print(classification_report(y_test, rf_pred))

    # Logistic Regression Model
    logr_model, logr_pred = model_results(
        name="logr",
        classifier=LogisticRegression(max_iter=131),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        write_dir=plot_dir,
    )

    print("Logistic Regression Results:\n")
    print(classification_report(y_test, logr_pred))

    # Random Forest Model
    knn_model, knn_pred = model_results(
        name="knn",
        classifier=KNeighborsClassifier(n_neighbors=20),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        write_dir=plot_dir,
    )

    print("KNN Classifier Results:\n")
    print(classification_report(y_test, knn_pred))

    # Models Results as html
    combine_html(
        combines={
            "<h2> Random Forest <h2>": f"{plot_dir}/rfc_cm.html",
            "<h3> RF Classification Report <h3>": f"{plot_dir}/rfc_cr.html",
            "<h2> Logistic Regression <h2>": f"{plot_dir}/logr_cm.html",
            "<h3> LR Classification Report <h3>": f"{plot_dir}/logr_cr.html",
            "<h2> KNN Classifier <h2>": f"{plot_dir}/knn_cm.html",
            "<h3> KNN Classification Report <h3>": f"{plot_dir}/knn_cr.html",
        },
        result="Output/Model_Results.html",
        head=f"{data_name} Models Report",
    )

    """
    With the three models trained, Random Forest, KNN Classifier, and Logistic Regression
    accuracies are closer to each other. Logistic Regression has slightly more accuracy.
    Looking at their confusion matrices, Logistic Regression Model predicts more home
    wins correctly than others.
    So, using these metrics, I would choose a basic Logistic Regression Model.
    Creating more good features and improving the models by cutting down garbage,
    may improve their performance in future.
    """

    return


if __name__ == "__main__":
    sys.exit(main())
