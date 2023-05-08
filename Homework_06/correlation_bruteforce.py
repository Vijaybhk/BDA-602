from __future__ import annotations

import sys
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import ndarray
from pandas import DataFrame, Series
from scipy import stats
from scipy.stats import pearsonr
from utilities import make_clickable
from variables import DfVariableProcessor


def fill_na(data: Series | Sequence | ndarray) -> Series | Sequence | ndarray:
    """
    Function to fill zeroes in na

    Parameters
    -----------
        data : Series | Sequence
            Input dataframe column to fill zeroes in place of na

    Returns
    -----------
        Series | Sequence
            Returns the input column after filling zeroes

    """
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cat_correlation(
    x: Series | ndarray,
    y: Series | ndarray,
    bias_correction: bool = True,
    tschuprow: bool = False,
) -> float:
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer 's V ( default )
    2. Tschuprow 's T

    SOURCES:
    1. CODE: https://github.com/MavericksDS/pycorr
    2. Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2

    Bias correction and formula's taken from :
    https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow 's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cont_correlation_ratio(
    categories: Series | ndarray, values: Series | ndarray
) -> float:
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


class CorrelationAndBruteForce(DfVariableProcessor):
    """
    Custom Class for Correlation And Brute Force.
    As Sub Class to DfVariableProcessor.
    """

    def __init__(
        self, input_df: DataFrame, predictors: list, response: str, write_dir: str
    ):
        """
        Parameters
        -------------
            input_df: DataFrame
                Input Dataframe, super class input

            predictors: list
                Input list of Predictors, super class input

            response: str
                Input response variable name, super class input

            write_dir: str
                Plots directory path to save plots and tables

        """
        super().__init__(input_df, predictors, response)
        self.write_dir = write_dir

    def corr_table_and_matrix(self, name: str, save: bool = True) -> DataFrame:
        """
        Parameters
        -------------
            name: str
                "Cat_Cat_Cramer", "Cat_Cat_Tschuprow", Cat_Cont", "Cont_Cont"
                Type of input names, to get correlation tables and matrices

            save: bool = True
                whether to save the correlation matrix and table

        Returns
        ----------------
            corr_df: DataFrame
                Correlation Dataframe with Variable pairs and Correlation coefficients.
                And links to difference with mean of response plots

        """
        cat_pred, cont_pred, _ = super().get_cat_and_cont_predictors()
        graph_list = []
        table_list = []

        splits = name.split("_")
        a = splits[0]
        b = splits[1]
        c = splits[-1]
        tschuprow = True if c == "Tschuprow" else False

        pred1 = cont_pred if a == "Cont" else cat_pred
        pred2 = cont_pred if b == "Cont" else cat_pred
        plot_title = "Correlation Matrix"
        typ = ""

        for i in pred1:
            for j in pred2:

                if a == "Cont" and b == "Cont":
                    res = pearsonr(self.input_df[i], self.input_df[j])[0]
                    plot_title = "Continuous / Continuous "
                    typ = "Pearson's r"

                elif a == "Cat" and b == "Cat":
                    res = cat_correlation(
                        self.input_df[i], self.input_df[j], tschuprow=tschuprow
                    )
                    if tschuprow:
                        plot_title = "Categorical / Categorical "
                        typ = "Tschuprow 's T"
                    else:
                        plot_title = "Categorical / Categorical "
                        typ = "Cramer's V"

                elif a == "Cat" and b == "Cont":
                    res = cat_cont_correlation_ratio(self.input_df[i], self.input_df[j])
                    plot_title = "Continuous / Categorical "
                    typ = "Correlation Ratio"

                else:
                    res = 0

                graph_list.append([i, j, res])

                if i != j:
                    inner_list = [
                        i,
                        j,
                        res,
                        f"./Plots/Combined_Diff_of__{i}.html",
                        f"./Plots/Combined_Diff_of__{j}.html",
                    ]
                    table_list.append(inner_list)

        corr_df = pd.DataFrame(
            table_list, columns=[f"{a}_1", f"{b}_2", "Corr", f"{a}_1_url", f"{b}_2_url"]
        )

        corr_df.sort_values(
            by=["Corr", f"{a}_1"], ascending=[False, True], inplace=True
        )

        if not save:

            return corr_df

        else:
            corr_df.to_html(
                f"{self.write_dir}/{name}_corr.html",
                formatters={
                    f"{a}_1_url": make_clickable,
                    f"{b}_2_url": make_clickable,
                },
                escape=False,
                index=False,
            )

            heat_df = pd.DataFrame(graph_list, columns=[f"{a}_1", f"{b}_2", "corr"])
            heat_df["text"] = [round(i, 6) for i in heat_df["corr"]]

            fig = go.Figure(
                go.Heatmap(
                    z=heat_df["corr"],
                    y=heat_df[f"{a}_1"],
                    x=heat_df[f"{b}_2"],
                    hoverongaps=False,
                    colorscale="RdBu",
                    zauto=True,
                    zmid=0,
                    text=heat_df["text"],
                    texttemplate="%{text}",
                )
            )

            fig.update_layout(
                title=f"<b>{plot_title} Correlation Matrix - {typ}</b>",
                xaxis={"title": "<b> Category 1 </b>"},
                yaxis_title="<b> Category 2 </b>",
                # height=500,
                paper_bgcolor="#fafafa",
            )

            fig.write_html(
                file=f"{self.write_dir}/{name}_matrix.html", include_plotlyjs="cdn"
            )

            return corr_df

    def get_all_correlation_metrics(self):
        """
        Saves all correlation tables and matrices for all input name types

        """

        for i in ["Cat_Cat_Cramer", "Cat_Cat_Tschuprow", "Cont_Cont", "Cat_Cont"]:
            self.corr_table_and_matrix(name=i, save=True)

        return

    def brute_force_matrix(self, name: str) -> DataFrame:
        """
        Parameters
        -------------
            name: str
                "Cat_Cat_Cramer", "Cat_Cat_Tschuprow", Cat_Cont", "Cont_Cont"
                Type of input names, to get correlation tables and matrices

        Returns
        ----------------
            brute_df: DataFrame
                Brute Force Dataframe with Variable pairs (with links to response plots)
                Correlation coefficients their absolute values.
                Links to brute force matrices

        """

        corr_df = self.corr_table_and_matrix(name=name, save=False)
        brute_lst = []
        splits = name.split("_")
        f1 = splits[0]
        f2 = splits[1]

        for a in range(len(corr_df)):
            p1 = corr_df.loc[a][0]
            p2 = corr_df.loc[a][1]

            df = self.input_df
            res = self.response
            pop_mean = df[res].mean()

            x1_mid = []
            x2_mid = []
            df_list = []

            if name == "Cont_Cont":
                nbins = 10
                _, x1 = pd.cut(x=df[p1], bins=nbins, retbins=True)
                _, x2 = pd.cut(x=df[p2], bins=nbins, retbins=True)
                x1_l = []
                x2_l = []
                x1_u = []
                x2_u = []

                for i in range(nbins):
                    x1_l.append(x1[i])
                    x1_u.append(x1[i + 1])
                    x1_mid.append(str((x1[i] + x1[i + 1]) / 2))

                    x2_l.append(x2[i])
                    x2_u.append(x2[i + 1])
                    x2_mid.append(str((x2[i] + x2[i + 1]) / 2))

                for j in range(nbins):
                    for k in range(nbins):
                        x1_cond = df[p1].between(x1_l[j], x1_u[j], inclusive="right")
                        x2_cond = df[p2].between(x2_l[k], x2_u[k], inclusive="right")

                        y_bin_response = df[x1_cond & x2_cond][res].mean()
                        y_bin_counts = df[x1_cond & x2_cond][res].count()

                        lst = [x1_mid[j], x2_mid[k], y_bin_response, y_bin_counts]
                        df_list.append(lst)

            elif name.startswith("Cat_Cat"):
                x1_mid = np.sort(df[p1].unique())
                x2_mid = np.sort(df[p2].unique())

                for j in x1_mid:
                    for k in x2_mid:

                        y_bin_response = df[(df[p1] == j) & (df[p2] == k)][res].mean()
                        y_bin_counts = df[(df[p1] == j) & (df[p2] == k)][res].count()

                        lst = [j, k, y_bin_response, y_bin_counts]
                        df_list.append(lst)

            elif name == "Cat_Cont":
                x1_mid = np.sort(df[p1].unique())
                nbins2 = 10
                _, x2 = pd.cut(x=df[p2], bins=nbins2, retbins=True)
                x2_l = []
                x2_u = []

                for i in range(nbins2):
                    x2_l.append(x2[i])
                    x2_u.append(x2[i + 1])
                    x2_mid.append(str((x2[i] + x2[i + 1]) / 2))

                for j in x1_mid:
                    for k in range(nbins2):
                        x2_cond = df[p2].between(x2_l[k], x2_u[k], inclusive="right")

                        y_bin_response = df[(df[p1] == j) & x2_cond][res].mean()
                        y_bin_counts = df[(df[p1] == j) & x2_cond][res].count()

                        lst = [j, x2_mid[k], y_bin_response, y_bin_counts]
                        df_list.append(lst)

            hdf = pd.DataFrame(df_list, columns=[p1, p2, "z", "pop"])
            tot = hdf["pop"].sum()
            hdf = hdf.fillna("")
            msd = 0
            w_msd = 0
            for i in range(len(hdf)):
                if hdf.loc[i][2] != "":
                    msd += (hdf.loc[i][2] - pop_mean) ** 2
                    w_msd += ((hdf.loc[i][2] - pop_mean) ** 2) * (hdf.loc[i][3] / tot)

            uw_msd = msd / len(hdf)

            inner_lst = [p1, p2, uw_msd, w_msd, f"./Plots/Brute__{p1}_{p2}.html"]
            brute_lst.append(inner_lst)

            text = []
            for c in range(len(hdf)):
                if hdf.loc[c][2] != "":
                    d = round(hdf.loc[c][2], 3)
                    e = hdf.loc[c][3]
                    text.append("{} (pop: {})".format(d, e))
                else:
                    text.append("")

            fig = go.Figure(
                go.Heatmap(
                    z=hdf["z"],
                    y=hdf[p2],
                    x=hdf[p1],
                    hoverongaps=False,
                    colorscale="RdBu",
                    zauto=True,
                    zmid=pop_mean,
                    text=text,
                    texttemplate="%{text}",
                )
            )

            fig.update_layout(
                title=f"<b> {p1} vs {p2} (Bin Averages) </b>",
                xaxis_title=f"<b> {p1} </b>",
                yaxis_title=f"<b> {p2} </b>",
            )

            fig.write_html(
                file=f"{self.write_dir}/Brute__{p1}_{p2}.html", include_plotlyjs="cdn"
            )

        brute_df = pd.DataFrame(
            brute_lst,
            columns=[
                f"{f1}_1",
                f"{f2}_2",
                "Unweighted_msd_ranking",
                "Weighted_msd_ranking",
                "Link",
            ],
        )

        brute_df = corr_df.merge(brute_df, on=[f"{f1}_1", f"{f2}_2"])
        brute_df.drop(columns=[f"{f1}_1", f"{f2}_2"], inplace=True)
        brute_df.rename(
            columns={f"{f1}_1_url": f"{f1}_1", f"{f2}_2_url": f"{f2}_2"}, inplace=True
        )
        abs_corr_lst = []
        if len(brute_df) > 0:
            for j in brute_df["Corr"]:
                if j >= 0:
                    abs_corr_lst.append(j)
                else:
                    abs_corr_lst.append(-j)
        brute_df["Abs_Corr"] = abs_corr_lst
        brute_df = brute_df[
            [
                f"{f1}_1",
                f"{f2}_2",
                "Unweighted_msd_ranking",
                "Weighted_msd_ranking",
                "Corr",
                "Abs_Corr",
                "Link",
            ]
        ]

        brute_df.sort_values(by=["Weighted_msd_ranking"], ascending=False, inplace=True)

        return brute_df

    def get_all_brute_force_metrics(self):
        """
        Saves all brute force tables and matrices for all input name types

        """

        for i in ["Cat_Cat", "Cont_Cont", "Cat_Cont"]:

            a, b = i.split("_")

            if i != "Cat_Cat":

                brute_df = self.brute_force_matrix(name=i)

            else:
                brute_1 = self.brute_force_matrix(name="Cat_Cat_Cramer")
                brute_2 = self.brute_force_matrix(name="Cat_Cat_Tschuprow")
                brute_df = brute_1.merge(
                    brute_2,
                    on=[
                        f"{a}_1",
                        f"{b}_2",
                        "Unweighted_msd_ranking",
                        "Weighted_msd_ranking",
                        "Link",
                    ],
                    suffixes=("_Cramer", "_Tschuprow"),
                )

                brute_df = brute_df[
                    [
                        f"{a}_1",
                        f"{b}_2",
                        "Unweighted_msd_ranking",
                        "Weighted_msd_ranking",
                        "Corr_Cramer",
                        "Corr_Tschuprow",
                        "Abs_Corr_Cramer",
                        "Abs_Corr_Tschuprow",
                        "Link",
                    ]
                ]

            brute_df.to_html(
                f"{self.write_dir}/{i}_brute.html",
                formatters={
                    f"{a}_1": make_clickable,
                    f"{b}_2": make_clickable,
                    "Link": make_clickable,
                },
                escape=False,
                index=False,
            )

        return


def main():
    help(CorrelationAndBruteForce)


if __name__ == "__main__":
    sys.exit(main())
