from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from pandas import DataFrame


def combine_html(one: str, two: str, three: str, optional: str | None = None):
    """
    Function to combine html files.

    Parameters
    -------------
        one: str
            First html file to combine
        two: str
            Second html file to combine
        three: str
            Resulting html file after combined
        optional: str | None = None
            Optional file to combine

    Returns
    ---------
        None

    """
    # Reading data from file1
    with open(one) as fp:
        data = fp.read()

    # Reading data from file2
    with open(two) as fp:
        data2 = fp.read()

    # Merging 2 files
    # To add the data of file2
    # from next line
    data += '<h2 align="right">[contd.]</h2>'
    data += "<hr>"
    data += data2

    if optional is not None:
        with open(optional) as fp:
            data3 = fp.read()
        data += '<h2 align="right">[contd.]</h2>'
        data += "<hr>"
        data += data3

    with open(three, "w") as fp:
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
    def create_plot_dir() -> str:
        """
        Creates a Plots directory within the file directory

        Returns
        ---------
            out_dir: str
                Plots directory path

        """
        this_dir = os.path.dirname(os.path.realpath(__file__))
        out_dir = "{}/Plots".format(this_dir)
        # if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

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
        )

        dist_plot.write_html(
            file=f"{write_dir}/Distribution_Plot_of_{name}.html",
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
        )
        violin_plot.write_html(
            file="{}/Violin_plot_of_{}.html".format(write_dir, name),
            include_plotlyjs="cdn",
        )
        return

    @staticmethod
    def diff_mean_response_plot_continuous_predictor(
        df: DataFrame, predictor: str, response: str, write_dir: str, nbins: int = 10
    ):
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

        Returns
        -------------
            None

        """

        _, x_bins = pd.cut(x=df[predictor], bins=nbins, retbins=True)

        x_lower = []
        x_upper = []
        x_mid_values = []
        y_bin_response = []
        y_bin_counts = []
        population_mean = df[response].mean()

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
                "MeanSquareDiff": [(y - population_mean) ** 2 for y in y_bin_response],
                "PopulationProportion": [y / t_count for y in y_bin_counts],
            },
            index=range(nbins),
        )

        diff_mean_df.iloc[:, :7].to_html(
            "{}/Unweighted_Diff_Table_of_{}.html".format(write_dir, predictor),
            na_rep="na",
        )

        diff_mean_df["MeanSquareDiffWeighted"] = (
            diff_mean_df["MeanSquareDiff"] * diff_mean_df["PopulationProportion"]
        )

        diff_mean_df.loc[
            "Sum", "MeanSquareDiff":"MeanSquareDiffWeighted"
        ] = diff_mean_df.sum()

        diff_mean_df.to_html(
            "{}/Weighted_Diff_Table_of_{}.html".format(write_dir, predictor),
            na_rep="",
            justify="left",
        )

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
        )

        # title
        fig.update_layout(
            title_text="Difference with Mean of Response: {} and {}".format(
                predictor, response
            )
        )

        fig.write_html(
            file="{}/Diff_Plot_{}_and_{}.html".format(write_dir, predictor, response),
            include_plotlyjs="cdn",
        )

        combine_html(
            optional="{}/Diff_plot_{}_and_{}.html".format(
                write_dir, predictor, response
            ),
            one="{}/Unweighted_Diff_Table_of_{}.html".format(write_dir, predictor),
            three="{}/Combined_Diff_of_{}.html".format(write_dir, predictor),
            two="{}/Weighted_Diff_Table_of_{}.html".format(write_dir, predictor),
        )

        return

    @staticmethod
    def diff_mean_response_plot_categorical_predictor(
        df: DataFrame, predictor: str, response: str, write_dir: str
    ):
        """
        Creates difference with mean of response plots for categorical predictors
        and saves as html files in the write directory.
        Also, saves weighted and unweighted mean of response dataframes as html.

        Parameters
        --------------

            df: DataFrame
                Input dataframe
            predictor: str
                predictor in the dataframe which is a class variable
            response: str
                predictor in dataframe which is a response variable
            write_dir: str
                Input the write directory path where the plots are to be saved

        Returns
        -------------
            None

        """
        population_mean = df[response].mean()
        x_uniques = df[predictor].unique()
        nbins = len(x_uniques)

        y_bin_response = []
        y_bin_counts = []

        for i in x_uniques:
            y_bin_response.append(df[df[predictor] == i][response].mean())
            y_bin_counts.append(df[df[predictor] == i][response].count())

        t_count = sum(y_bin_counts)

        diff_mean_df = pd.DataFrame(
            {
                "Bins": x_uniques,
                "BinCount": y_bin_counts,
                "BinMeans": y_bin_response,
                "PopulationMean": population_mean,
                "MeanSquareDiff": [(y - population_mean) ** 2 for y in y_bin_response],
                "PopulationProportion": [y / t_count for y in y_bin_counts],
            },
            index=range(nbins),
        )

        diff_mean_df.iloc[:, :5].to_html(
            "{}/Unweighted_Diff_Table_of_{}.html".format(write_dir, predictor),
            na_rep="na",
        )

        diff_mean_df["MeanSquareDiffWeighted"] = (
            diff_mean_df["MeanSquareDiff"] * diff_mean_df["PopulationProportion"]
        )

        diff_mean_df.loc[
            "Sum", "MeanSquareDiff":"MeanSquareDiffWeighted"
        ] = diff_mean_df.sum()

        diff_mean_df.to_html(
            "{}/Weighted_Diff_Table_of_{}.html".format(write_dir, predictor),
            na_rep="",
            justify="left",
        )

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=df[predictor],
                y=df[response],
                name="Population",
                yaxis="y2",
                opacity=0.5,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_uniques,
                y=y_bin_response,
                name="Bin Mean",
                yaxis="y",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_uniques,
                y=[population_mean] * len(x_uniques),
                name="Population Mean",
                yaxis="y",
            )
        )

        # axes objects
        fig.update_layout(
            xaxis=dict(title="Predictor Bin"),
            # 1st y axis
            yaxis=dict(
                title="Response",
                titlefont=dict(color="#1f77b4"),
                tickfont=dict(color="#1f77b4"),
            ),
            # 2nd y axis
            yaxis2=dict(title="Population", overlaying="y", side="right"),
            legend=dict(x=1.1, y=1),
        )

        # title
        fig.update_layout(
            title_text="Difference with Mean of Response Plot for {} and {}".format(
                predictor, response
            ),
        )

        fig.write_html(
            file="{}/Diff_Plot_{}_and_{}.html".format(write_dir, predictor, response),
            include_plotlyjs="cdn",
        )

        combine_html(
            optional="{}/Diff_plot_{}_and_{}.html".format(
                write_dir, predictor, response
            ),
            one="{}/Unweighted_Diff_Table_of_{}.html".format(write_dir, predictor),
            three="{}/Combined_Diff_of_{}.html".format(write_dir, predictor),
            two="{}/Weighted_Diff_Table_of_{}.html".format(write_dir, predictor),
        )

        return

    def cat_response_cont_predictor(
        self, cat_resp: str, cont_pred: str, write_dir: str
    ):
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
            None

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

        self.diff_mean_response_plot_continuous_predictor(
            df=self.input_df,
            predictor=cont_pred,
            response=cat_resp,
            write_dir=write_dir,
        )

        combine_html(
            one="{}/Violin_plot_of_{}.html".format(write_dir, cont_pred),
            two="{}/Distribution_plot_of_{}.html".format(write_dir, cont_pred),
            three="{}/Combined_plot_of_{}.html".format(write_dir, cont_pred),
        )

        return

    def cont_response_cat_predictor(
        self, cont_resp: str, cat_pred: str, write_dir: str
    ):
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
            None

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

        self.diff_mean_response_plot_categorical_predictor(
            df=self.input_df,
            predictor=cat_pred,
            response=cont_resp,
            write_dir=write_dir,
        )

        combine_html(
            one="{}/Violin_plot_of_{}.html".format(write_dir, cat_pred),
            two="{}/Distribution_plot_of_{}.html".format(write_dir, cat_pred),
            three="{}/Combined_plot_of_{}.html".format(write_dir, cat_pred),
        )

        return

    def cat_response_cat_predictor(self, cat_resp: str, cat_pred: str, write_dir: str):
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
            None

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
            file="{}/Density_Heat_Map_of_{}.html".format(write_dir, cat_pred),
            include_plotlyjs="cdn",
        )

        self.diff_mean_response_plot_categorical_predictor(
            df=self.input_df,
            predictor=cat_pred,
            response=cat_resp,
            write_dir=write_dir,
        )

        return

    def cont_response_cont_predictor(
        self, cont_resp: str, cont_pred: str, write_dir: str
    ):
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
            None

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
            file="{}/scatter_plot_of_{}.html".format(write_dir, cont_pred),
            include_plotlyjs="cdn",
        )

        self.diff_mean_response_plot_continuous_predictor(
            df=self.input_df,
            predictor=cont_pred,
            response=cont_resp,
            write_dir=write_dir,
        )

        return

    def get_all_plots(
        self,
        cont_pred: list[str],
        cat_pred: list[str],
        response: str,
        res_type: str,
        write_dir: str,
    ) -> tuple[dict, dict]:
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

        """

        # Two dicts for predictor plots and mean of response plots
        diff_dict = {}
        plot_dict = {}

        # Loops to execute plots for particular predictor and response types.
        # Also, to store paths to diff dict and plot dict.
        for i in cont_pred:
            if res_type == "categorical":
                self.cat_response_cont_predictor(response, i, write_dir)
                diff_dict[i] = "./Plots/Combined_Diff_of_{}.html".format(i)
                plot_dict[i] = "./Plots/Combined_plot_of_{}.html".format(i)

            elif res_type == "continuous":
                self.cont_response_cont_predictor(response, i, write_dir)
                diff_dict[i] = "./Plots/Combined_Diff_of_{}.html".format(i)
                plot_dict[i] = "./Plots/scatter_plot_of_{}.html".format(i)

        for j in cat_pred:
            if res_type == "categorical":
                self.cat_response_cat_predictor(response, j, write_dir)
                diff_dict[j] = "./Plots/Combined_Diff_of_{}.html".format(j)
                plot_dict[j] = "./Plots/Density_Heat_Map_of_{}.html".format(j)

            elif res_type == "continuous":
                self.cont_response_cat_predictor(response, j, write_dir)
                diff_dict[j] = "./Plots/Combined_Diff_of_{}.html".format(j)
                plot_dict[j] = "./Plots/Combined_plot_of_{}.html".format(j)

        return diff_dict, plot_dict


def main():
    help(VariablePlotter)


if __name__ == "__main__":
    sys.exit(main())
