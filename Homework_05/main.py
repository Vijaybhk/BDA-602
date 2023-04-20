from __future__ import annotations

import os
import sys

from correlation_bruteforce import CorrelationAndBruteForce
from plotter import combine_html
from utilities import mariadb_df, predictor_reports


def main():
    # Name of the data set name to pass in as "heading of the html"
    data_name = "Baseball"

    try:
        # Getting data from baseball database and created features table
        query = "SELECT * FROM" + " features"
        df = mariadb_df(query=query)

        # Getting predictors and response from dataframe
        predictors = df.columns[2:-1]
        response = df.columns[-1]

        # Plots directory path
        this_dir = os.path.dirname(os.path.realpath(__file__))
        plot_dir = f"{this_dir}/Plots"
        os.makedirs(plot_dir, exist_ok=True)

        # Creating predictor reports as html
        # Gets the list of cat and cont predictors
        cat_pred, cont_pred = predictor_reports(
            df=df, predictors=predictors, response=response, plot_dir=plot_dir
        )

        # Instantiating CorrelationAndBruteForce class with inputs
        corr_bf = CorrelationAndBruteForce(
            input_df=df, predictors=predictors, response=response, write_dir=plot_dir
        )

        # Gets all correlation, brute force tables and matrices as html
        corr_bf.get_all_correlation_metrics()
        corr_bf.get_all_brute_force_metrics()

        # Empty dictionaries
        cat_rep, cat_corr, cat_brut, cont_corr, cont_brut, cont_rep = (
            {},
            {},
            {},
            {},
            {},
            {},
        )

        # Headings of html as key
        # Relative path of html as value

        # If there is at least one Categorical predictor, gets all cat dictionaries
        if len(cat_pred) > 0:
            cat_corr = {
                "<h2> Categorical/Categorical Correlations </h2>"
                "<h3> Correlation Tschuprow Matrix </h3>": "./Plots/Cat_Cat_Tschuprow_matrix.html",
                "<h3> Correlation Cramer Matrix </h3>": "./Plots/Cat_Cat_Cramer_matrix.html",
                "<div class='row'> <div class='column'>"
                "<h3> Correlation Tschuprow Table </h3>": "./Plots/Cat_Cat_Tschuprow_corr.html",
                "</div> <div class='column'>"
                " <h3> Correlation Cramer Table </h3>": "./Plots/Cat_Cat_Cramer_corr.html",
                "</div> <h2> Categorical/Continuous Correlations </h2>"
                "<h3> Correlation Ratio Matrix </h3>": "./Plots/Cat_Cont_matrix.html",
                "<h3> Correlation Ratio Table </h3>": "./Plots/Cat_Cont_corr.html",
            }

            cat_brut = {
                "<h2> Categorical/Categorical Brute Force </h2>": "./Plots/Cat_Cat_brute.html",
                "<h2> Categorical/Continuous Brute Force </h2>": "./Plots/Cat_Cont_brute.html",
            }

            cat_rep = {
                "<h2> Categorical Predictors </h2>": "./Plots/report_categorical.html"
            }

        # If there is at least one Continuous predictor, gets all cont dictionaries
        if len(cont_pred) > 0:
            cont_corr = {
                "<h2> Continuous/Continuous Correlations </h2>"
                "<h3> Correlation Pearson's Matrix </h3>": "./Plots/Cont_Cont_matrix.html",
                "<h3> Correlation Pearson's Table </h3>": "./Plots/Cont_Cont_corr.html",
            }

            cont_brut = {
                "<h2> Continuous/Continuous Brute Force </h2>": "./Plots/Cont_Cont_brute.html"
            }

            cont_rep = {
                "<h2> Continuous Predictors </h2>": "./Plots/report_continuous.html"
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
            result="report.html",
            head=data_name,
        )

    except Exception as err:
        print(type(err))
        print("Unable to connect to Mariadb")

    return


if __name__ == "__main__":
    sys.exit(main())
