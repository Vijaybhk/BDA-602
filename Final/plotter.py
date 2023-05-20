from __future__ import annotations

import sys

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from pandas import DataFrame


def combine_html(combines: dict, result: str, head: str = ""):
    """
    Function to combine html files.

    Parameters
    -------------
        combines: dict
            Dictionary of html files to be combined
            header of the html file as key
            path of the html file as value

        result: str
            Result html file path with file name as str

        head: str
            Header to the resulting html file as str

    Returns
    ---------
        None

    """

    data = (
        """
        <meta charset="utf-8" />
        <style>
        table {border-collapse: collapse; margin-left: auto; margin-right: auto;}
        tr:nth-child(even) {background-color: #f2f2f2;}
        th, td {padding: 15px; font-size: 14pt;}
        th {background-color: #d0dbe6; text-align: center;}
        body {
        font-family: "Helvetica", "Arial", "sans-serif";
        font-size: 14pt;
        background-color: #fafafa;
        }
        h1, h2, h3, h4, h5, h6 {text-align: center;}
        a {color: #18646d;}
        a:visited {color: #5a144e;}
        /* Create a two-column layout */
        .column {float: left; width: 50%;}
        /* Clear floats after the columns */
        .row:after {content: ""; display: table; clear: both;}
        </style><br>
        """
        + f"<h1>{head.title()}</h1>\n"
    )

    for header in combines.keys():
        # Reading data from a file
        file = combines[header]
        with open(file) as fp:
            data = data + f"{header}\n" + fp.read()
        data = data + "\n<br>"

    with open(result, "w") as fp:
        # Writing the final html
        fp.write(data)

    return


class VariablePlotter:
    """
    Custom Class to Plot Variables in a Dataframe
    """

    def __init__(self, input_df: DataFrame | None = None):
        """
        Constructor Method for Plotter Class

        Parameters
        -------------
            input_df: DataFrame | None = None
                Input DataFrame

        """
        self.input_df = input_df

    @staticmethod
    def distribution_plot(
        df: DataFrame, cat_var: str, cont_var: str, write_dir: str, restype: str
    ):
        """
        Static Method to Creates Distribution Plots for Cat vs Cont Variables
        Used Within this class

        Parameters
        ------------

            df: DataFrame
                Input Dataframe
            cat_var: str
                Categorical Variable(Response/Predictor)
            cont_var: str
                Continuous Variable(Response/Predictor)
            write_dir: str
                Directory to save plots
            restype: str
                Response Type "cat" or "cont"

        Returns
        -------------
            None

        """
        hist_data = []
        group_labels = []

        for j in df[cat_var].unique():
            x = df[df[cat_var] == j][cont_var]
            hist_data.append(x)
            group_labels.append(f"{cat_var} = " + str(j))

        bin_size = (df[cont_var].max() - df[cont_var].min()) / 15

        if restype == "cat":
            name = cont_var
            x_axis = "Predictor"
        else:
            name = cat_var
            x_axis = "Response"

        dist_plot = ff.create_distplot(
            hist_data=hist_data, group_labels=group_labels, bin_size=bin_size
        )

        dist_plot.update_layout(
            title=f"Distribution Plot of {name}",
            xaxis_title=f"{x_axis} : {cont_var}",
            yaxis_title="Distribution",
            paper_bgcolor="#fafafa",
        )

        dist_plot.write_html(
            file=f"{write_dir}/Distribution_plot_of_{name}.html",
            include_plotlyjs="cdn",
        )

        return

    @staticmethod
    def violin_plot(
        df: DataFrame, cat_var: str, cont_var: str, write_dir: str, restype: str
    ):
        """
        Static Method to Creates Violin Plots for Cat vs Cont Variables
        Used Within this class

        Parameters
        -------------

            df: DataFrame
                Input Dataframe
            cat_var: str
                Categorical Variable(Response/Predictor)
            cont_var: str
                Continuous Variable(Response/Predictor)
            write_dir: str
                Directory to save plots
            restype: str
                Response Type "cat" or "cont"

        Returns
        -------------
            None

        """
        if restype == "cat":
            name = cont_var
            x_axis = "Response"
            y_axis = "Predictor"
        else:
            name = cat_var
            x_axis = "Predictor"
            y_axis = "Response"

        violin_plot = px.violin(
            data_frame=df,
            y=cont_var,
            x=cat_var,
            box=True,
            color=cat_var,
        )
        violin_plot.update_layout(
            title="Violin Plot of {}".format(name),
            xaxis_title="{} : {}".format(x_axis, cat_var),
            yaxis_title="{} : {}".format(y_axis, cont_var),
            width=1300,
            height=700,
            paper_bgcolor="#fafafa",
        )
        violin_plot.write_html(
            file="{}/Violin_plot_of_{}.html".format(write_dir, name),
            include_plotlyjs="cdn",
        )
        return

    @staticmethod
    def diff_mean_response_plots(
        df: DataFrame,
        predictor: str,
        response: str,
        write_dir: str,
        predtype: str,
        nbins: int = 10,
    ) -> tuple[float, float]:
        """
        Creates difference with mean of response plots for numerical/continuous predictors
        and saves as html files in the write directory.
        Also, saves weighted and unweighted mean of response dataframes as html

        Parameters
        ---------------

            df: DataFrame
                Input dataframe
            predictor: str
                predictor column in the dataframe which is a numerical/continuous variable
            response: str
                response variable
            write_dir: str
                Input the write directory path where the plots are to be saved
            nbins: int = 10 as default
                Number of bins to be divided in the bar plot, default is 10.
            predtype: str = None as default
                Predictors Type either categorical or continuous

        Returns
        -------------
            msd: tuple[float, float]
                tuple of Mean Square Difference Unweighted and Weighted

        """

        x_lower = []
        x_upper = []
        x_mid_values = []
        y_bin_response = []
        y_bin_counts = []
        population_mean = df[response].mean()

        if predtype == "continuous":
            _, x_bins = pd.cut(x=df[predictor], bins=nbins, retbins=True)

            for i in range(nbins):
                x_lower.append(x_bins[i])
                x_upper.append(x_bins[i + 1])
                x_mid_values.append((x_lower[i] + x_upper[i]) / 2)

                # x range Inclusive on the right side/upper limit
                y_bin_response.append(
                    df[(df[predictor] > x_bins[i]) & (df[predictor] <= x_bins[i + 1])][
                        response
                    ].mean()
                )

                y_bin_counts.append(
                    df[(df[predictor] > x_bins[i]) & (df[predictor] <= x_bins[i + 1])][
                        response
                    ].count()
                )

            t_count = sum(y_bin_counts)

            diff_mean_df = pd.DataFrame(
                {
                    "LowerBin": x_lower,
                    "UpperBin": x_upper,
                    "BinCenters": x_mid_values,
                    "BinCount": y_bin_counts,
                    "BinMeans": y_bin_response,
                    "PopulationMean": [population_mean] * nbins,
                    "MeanSquareDiff": [
                        (y - population_mean) ** 2 for y in y_bin_response
                    ],
                    "PopulationProportion": [y / t_count for y in y_bin_counts],
                },
                index=range(nbins),
            )

        else:
            x_mid_values = df[predictor].unique()
            nbins = len(x_mid_values)
            for i in x_mid_values:
                y_bin_response.append(df[df[predictor] == i][response].mean())
                y_bin_counts.append(df[df[predictor] == i][response].count())

            t_count = sum(y_bin_counts)

            diff_mean_df = pd.DataFrame(
                {
                    "Bins": x_mid_values,
                    "BinCount": y_bin_counts,
                    "BinMeans": y_bin_response,
                    "PopulationMean": population_mean,
                    "MeanSquareDiff": [
                        (y - population_mean) ** 2 for y in y_bin_response
                    ],
                    "PopulationProportion": [y / t_count for y in y_bin_counts],
                },
                index=range(nbins),
            )

        diff_mean_df["MeanSquareDiffUnWeighted"] = (
            diff_mean_df["MeanSquareDiff"] / nbins
        )

        diff_mean_df["MeanSquareDiffWeighted"] = (
            diff_mean_df["MeanSquareDiff"] * diff_mean_df["PopulationProportion"]
        )

        msd = (
            diff_mean_df["MeanSquareDiffUnWeighted"].sum(),
            diff_mean_df["MeanSquareDiffWeighted"].sum(),
        )

        diff_mean_df.loc[
            "sum", "MeanSquareDiff":"MeanSquareDiffWeighted"
        ] = diff_mean_df.sum()

        df_path = "{}/Weighted_Diff_Table_of_{}.html".format(write_dir, predictor)
        diff_mean_df.to_html(df_path, na_rep="", justify="left")

        # Diff With Mean of Response Plot
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=x_mid_values,
                y=y_bin_counts,
                name="Population",
                yaxis="y2",
                opacity=0.5,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_mid_values,
                y=y_bin_response,
                name="Bin Mean(μ𝑖)",
                yaxis="y",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_mid_values,
                y=[population_mean] * nbins,
                name="Population Mean(μpop)",
                yaxis="y",
                mode="lines",
            )
        )

        # axes objects
        fig.update_layout(
            xaxis=dict(title="Predictor Bin"),
            # 1st y axis
            yaxis=dict(title="Response"),
            # 2nd y axis
            yaxis2=dict(title="Population", overlaying="y", side="right"),
            legend=dict(x=1, y=1),
            paper_bgcolor="#fafafa",
        )

        # title
        fig.update_layout(title_text="{} and {}".format(predictor, response))

        fig_path = "{}/Diff_Plot_{}_and_{}.html".format(write_dir, predictor, response)
        fig.write_html(file=fig_path, include_plotlyjs="cdn")

        combine_html(
            combines={
                "<h3> Difference with Mean of Response Plot </h3>": f"{fig_path}",
                "<h3> Difference with Mean of Response Table </h3>": f"{df_path}",
            },
            result=f"{write_dir}/Combined_Diff_of__{predictor}.html",
        )

        return msd

    def cat_response_cont_predictor(
        self, cat_resp: str, cont_pred: str, write_dir: str
    ) -> tuple[float, float]:
        """
        Method to Create Plots for Categorical Response and Continuous Predictor
        and also get Difference in Mean of Response plots and Tables for the Predictor

        Parameters
        -------------

            cat_resp: str
                Categorical Response Variable
            cont_pred: str
                Continuous Predictor Variable
            write_dir: str
                Directory to Save the plots

        Returns
        -------------
            msd: tuple[float, float]
                tuple of Mean Square Difference Unweighted and Weighted

        """

        self.violin_plot(
            df=self.input_df,
            cat_var=cat_resp,
            cont_var=cont_pred,
            write_dir=write_dir,
            restype="cat",
        )

        self.distribution_plot(
            df=self.input_df,
            cat_var=cat_resp,
            cont_var=cont_pred,
            write_dir=write_dir,
            restype="cat",
        )

        msd = self.diff_mean_response_plots(
            df=self.input_df,
            predictor=cont_pred,
            response=cat_resp,
            write_dir=write_dir,
            predtype="continuous",
        )

        combine_html(
            combines={
                "<h3> Violin Plot </h3>": "{}/Violin_plot_of_{}.html".format(
                    write_dir, cont_pred
                ),
                "<h3> Distribution Plot </h3>": "{}/Distribution_plot_of_{}.html".format(
                    write_dir, cont_pred
                ),
            },
            result="{}/Combined_plot_of__{}.html".format(write_dir, cont_pred),
        )

        return msd

    def cont_response_cat_predictor(
        self, cont_resp: str, cat_pred: str, write_dir: str
    ) -> tuple[float, float]:
        """
        Method to Create Plots for Continuous Response and Categorical Predictor
        and also get Difference in Mean of Response plots and Tables for the Predictor

        Parameters
        ------------

            cont_resp: str
                Continuous Response Variable
            cat_pred: str
                Categorical Predictor Variable
            write_dir: str
                Directory to Save the plots
        Returns
        -------------
            msd: tuple[float, float]
                tuple of Mean Square Difference Unweighted and Weighted

        """
        self.violin_plot(
            df=self.input_df,
            cat_var=cat_pred,
            cont_var=cont_resp,
            write_dir=write_dir,
            restype="cont",
        )

        self.distribution_plot(
            df=self.input_df,
            cat_var=cat_pred,
            cont_var=cont_resp,
            write_dir=write_dir,
            restype="cont",
        )

        msd = self.diff_mean_response_plots(
            df=self.input_df,
            predictor=cat_pred,
            response=cont_resp,
            write_dir=write_dir,
            predtype="categorical",
        )

        combine_html(
            combines={
                "<h3> Violin Plot </h3>": "{}/Violin_plot_of_{}.html".format(
                    write_dir, cat_pred
                ),
                "<h3> Distribution Plot </h3>": "{}/Distribution_plot_of_{}.html".format(
                    write_dir, cat_pred
                ),
            },
            result="{}/Combined_plot_of__{}.html".format(write_dir, cat_pred),
        )

        return msd

    def cat_response_cat_predictor(
        self, cat_resp: str, cat_pred: str, write_dir: str
    ) -> tuple[float, float]:
        """
        Method to Create Heat Density Plot for Categorical Response and Categorical Predictor
        and also get Difference in Mean of Response plots and Tables for the Predictor

        Parameters
        ------------

            cat_resp: str
                Categorical Response Variable
            cat_pred: str
                Categorical Predictor Variable
            write_dir: str
                Directory to Save the plots

        Returns
        -------------
            msd: tuple[float, float]
                tuple of Mean Square Difference Unweighted and Weighted

        """

        heat_plot = px.density_heatmap(
            data_frame=self.input_df,
            x=cat_pred,
            y=cat_resp,
            color_continuous_scale=px.colors.sequential.Viridis,
            text_auto=True,
        )

        heat_plot.update_layout(
            title="Heat Map of {}".format(cat_pred),
            xaxis_title="Predictor: {}".format(cat_pred),
            yaxis_title="Response: {}".format(cat_resp),
        )

        heat_plot.write_html(
            file="{}/Density_Heat_Map_of__{}.html".format(write_dir, cat_pred),
            include_plotlyjs="cdn",
        )

        msd = self.diff_mean_response_plots(
            df=self.input_df,
            predictor=cat_pred,
            response=cat_resp,
            write_dir=write_dir,
            predtype="categorical",
        )

        return msd

    def cont_response_cont_predictor(
        self, cont_resp: str, cont_pred: str, write_dir: str
    ) -> tuple[float, float]:
        """
        Method to Create Scatter Plot for Continuous Response and Continuous Predictor
        and also get Difference in Mean of Response plots and Tables for the Predictor

        Parameters
        -----------
            cont_resp: str
                Continuous Response Variable
            cont_pred: str
                Continuous Predictor Variable
            write_dir: str
                Directory to Save the plots

        Returns
        -------------
            msd: tuple[float, float]
                tuple of Mean Square Difference Unweighted and Weighted

        """

        scatter_plot = px.scatter(
            x=self.input_df[cont_pred], y=self.input_df[cont_resp], trendline="ols"
        )

        scatter_plot.update_layout(
            title="Scatter Plot of {}".format(cont_pred),
            xaxis_title="Predictor: {}".format(cont_pred),
            yaxis_title="Response: {}".format(cont_resp),
        )

        scatter_plot.write_html(
            file="{}/scatter_plot_of__{}.html".format(write_dir, cont_pred),
            include_plotlyjs="cdn",
        )

        msd = self.diff_mean_response_plots(
            df=self.input_df,
            predictor=cont_pred,
            response=cont_resp,
            write_dir=write_dir,
            predtype="continuous",
        )

        return msd

    def get_all_plots(
        self,
        cont_pred: list[str],
        cat_pred: list[str],
        response: str,
        res_type: str,
        write_dir: str,
    ) -> tuple[dict, dict, dict, dict]:
        """
        Method to get all the plots based on Continuous/Categorical Variables(Predictor/Response)
        Uses other methods in the class to generated outputs and save plots.

        Parameters
        ------------
            cont_pred: list[str]
                List of Continuous predictors you want plotted
            cat_pred: list[str]
                List of Categorical predictors you want plotted
            response: str
                Response Variable you want plotted
            res_type: str
                Response Type either "categorical" or "continuous"
            write_dir: str
                Path to the directory where you want all the plots stored

        Returns
        ------------
            diff_dict: dict
                Dictionary object with Variable name as key and combined path of
                difference in mean of response plots and tables

            plot_dict: dict
                Dictionary object with Variable name as key and combined path of
                Predictor vs Response plots

            uw_dict: dict
                Dictionary object with Variable name as key and Unweighted mean of response

            w_dict: dict
                Dictionary object with Variable name as key and Weighted mean of response

        """

        # Two dicts for predictor plots and mean of response plots
        diff_dict = {}
        plot_dict = {}
        uw_dict = {}
        w_dict = {}

        # Loops to execute plots for particular predictor and response types.
        # Also, to store paths to diff dict and plot dict.
        for i in cont_pred:
            if res_type == "categorical":
                msd = self.cat_response_cont_predictor(response, i, write_dir)
                diff_dict[i] = "./Plots/Combined_Diff_of__{}.html".format(i)
                plot_dict[i] = "./Plots/Combined_plot_of__{}.html".format(i)
                w_dict[i] = msd[1]
                uw_dict[i] = msd[0]

            elif res_type == "continuous":
                msd = self.cont_response_cont_predictor(response, i, write_dir)
                diff_dict[i] = "./Plots/Combined_Diff_of__{}.html".format(i)
                plot_dict[i] = "./Plots/scatter_plot_of__{}.html".format(i)
                w_dict[i] = msd[1]
                uw_dict[i] = msd[0]

        for j in cat_pred:
            if res_type == "categorical":
                msd = self.cat_response_cat_predictor(response, j, write_dir)
                diff_dict[j] = "./Plots/Combined_Diff_of__{}.html".format(j)
                plot_dict[j] = "./Plots/Density_Heat_Map_of__{}.html".format(j)
                w_dict[j] = msd[1]
                uw_dict[j] = msd[0]

            elif res_type == "continuous":
                msd = self.cont_response_cat_predictor(response, j, write_dir)
                diff_dict[j] = "./Plots/Combined_Diff_of__{}.html".format(j)
                plot_dict[j] = "./Plots/Combined_plot_of__{}.html".format(j)
                w_dict[j] = msd[1]
                uw_dict[j] = msd[0]

        return diff_dict, plot_dict, uw_dict, w_dict


def main():
    help(VariablePlotter)


if __name__ == "__main__":
    sys.exit(main())
