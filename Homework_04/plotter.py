import os

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


class VariablePlotter:
    def __init__(self, input_df=None):
        self.input_df = input_df

    @staticmethod
    def create_plot_dir():
        this_dir = os.path.dirname(os.path.realpath(__file__))
        out_dir = "{}/Plots".format(this_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return out_dir

    @staticmethod
    def distribution_plot(df, cat_var, cont_var, write_dir, restype):
        hist_data = []
        group_labels = []

        for j in df[cat_var].unique():
            x = df[df[cat_var] == j][cont_var]
            hist_data.append(x)
            group_labels.append(f"{cat_var} = " + str(j))

        bin_size = (df[cont_var].max() - df[cont_var].min()) / 15

        name = cont_var if restype == "cat" else cat_var

        dist_plot = ff.create_distplot(
            hist_data=hist_data, group_labels=group_labels, bin_size=bin_size
        )

        dist_plot.update_layout(
            title="Distribution Plot of {}".format(name),
            xaxis_title="{}".format(cont_var),
            yaxis_title="Distribution",
        )

        dist_plot.write_html(
            file="{}/Distribution Plot of {}.html".format(write_dir, name),
            include_plotlyjs="cdn",
        )

        return

    @staticmethod
    def violin_plot(df, cat_var, cont_var, write_dir, restype):

        name = cont_var if restype == "cat" else cat_var

        violin_plot = px.violin(
            data_frame=df,
            y=cont_var,
            x=cat_var,
            box=True,
            color=cat_var,
        )
        violin_plot.update_layout(
            title="Violin Plot of {}".format(name),
            xaxis_title="{}".format(cat_var),
            yaxis_title="{}".format(cont_var),
        )
        violin_plot.write_html(
            file="{}/Violin plot of {}.html".format(write_dir, name),
            include_plotlyjs="cdn",
        )
        return

    @staticmethod
    def diff_mean_response_plot_continuous_predictor(
        df, predictor, response, write_dir, nbins=10
    ):
        """
        Creates difference with mean of response plots for numerical/continuous predictors
        and saves as html files in the write directory
        :param df: Input dataframe
        :param predictor: predictor column in the dataframe which is a numerical/continuous variable
        :param response: response variable
        :param write_dir: Input the write directory path where the plots are to be saved
        :param nbins: Number of bins to be divided in the bar plot, default is 10.
        :return: A Dataframe with difference with mean of response table columns
        """

        # min_range = df[predictor].min()
        # max_range = df[predictor].max()
        # print(min_range, max_range, nbins)

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

        diff_mean_df = pd.DataFrame(
            {
                "LowerBin": x_lower,
                "UpperBin": x_upper,
                "BinCenters": x_mid_values,
                "BinCount": y_bin_counts,
                "BinMeans(μ𝑖)": y_bin_response,
                "PopulationMean(μ𝑝𝑜𝑝)": [population_mean] * nbins,
            },
            index=range(nbins),
        )

        # print(x_values)
        # print(x_ranges)
        # print(y_bin_response)
        # print(y_bin_counts)
        # print(out_df)

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
            file="{}/Diff Plot {} and {}.html".format(write_dir, predictor, response),
            include_plotlyjs="cdn",
        )

        return diff_mean_df

    def cat_response_cont_predictor(self, cat_resp, cont_pred, write_dir):

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

        return

    def cont_response_cat_predictor(self, cont_resp, cat_pred, write_dir):
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

        return

    def cat_response_cat_predictor(self, cat_resp, cat_pred, write_dir):

        heat_plot = px.density_heatmap(
            data_frame=self.input_df,
            x=cat_pred,
            y=cat_resp,
            range_color=[5, 8],
            # color_continuous_scale=px.colors.sequential.Viridis,
            text_auto=True,
        )

        heat_plot.update_layout(
            title="Heat Map of {}".format(cat_pred),
            xaxis_title="Predictor: {}".format(cat_pred),
            yaxis_title="Response: {}".format(cat_resp),
        )

        heat_plot.write_html(
            file="{}/Density Heat Map of {}.html".format(write_dir, cat_pred),
            include_plotlyjs="cdn",
        )

        return

    def cont_response_cont_predictor(self, cont_resp, cont_pred, write_dir):
        scatter_plot = px.scatter(
            x=self.input_df[cont_pred], y=self.input_df[cont_resp], trendline="ols"
        )

        scatter_plot.update_layout(
            title="Scatter Plot of {}".format(cont_pred),
            xaxis_title="Predictor: {}".format(cont_pred),
            yaxis_title="Response: {}".format(cont_resp),
        )

        scatter_plot.write_html(
            file="{}/scatter plot of {}.html".format(write_dir, cont_pred),
            include_plotlyjs="cdn",
        )

        self.diff_mean_response_plot_continuous_predictor(
            df=self.input_df,
            predictor=cont_pred,
            response=cont_resp,
            write_dir=write_dir,
        )

        return
