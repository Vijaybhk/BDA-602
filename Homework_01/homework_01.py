import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def dataframe_to_array(dataframe, a, b):
    """
    Converting dataframe to numpy array
    :param dataframe: input dataframe
    :param a: start number of the columns
    :param b: end number of the columns
    :return: numpy array with all the rows and columns with indexed columns
    """
    return dataframe.iloc[:, a:b].to_numpy()


def numpy_statistics(array, column_list):
    """
    Calculate Summary Statistics using NumPy
    :param array: Input array for which statistics are to be calculated
    :param column_list: List of column names for which statistics are to be calculated
    :return: an output dataframe with indexes as statistics and columns as column names of the primary dataframe
    """
    mean = np.mean(array, axis=0)
    minimum = np.min(array, axis=0)
    maximum = np.max(array, axis=0)
    quantiles = np.quantile(array, [0.25, 0.5, 0.75], axis=0)
    out_arr = np.vstack((mean, minimum, maximum, quantiles))
    out_df = pd.DataFrame(
        data=out_arr,
        index=["mean", "min", "max", "quartile", "median", "third quartile"],
        columns=column_list,
    )
    return out_df


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


def main():
    # Using URL for data instead of downloading for better reproducibility
    data_path = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    )

    # Data from source does not have column names included
    col_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]

    # Loading dataset into a pandas dataframe
    df = pd.read_csv(data_path, header=None, names=col_names)

    # Checking head and tail of the dataframe
    print(df.head())
    print(df.tail())

    # Summary Statistics using Pandas dataframe describe function
    print(df.describe())

    # Converting the dataframe df to numpy array
    np_array = dataframe_to_array(df, 0, 4)

    # Summary statistics using numpy
    stat_df = numpy_statistics(np_array, col_names[:-1])
    print(stat_df)

    # Check Number of Missing Values in each column
    # Found to be None
    print(df.isna().sum())

    # Using Standard Scaler
    std_scaler = StandardScaler()
    x = df.iloc[:, :-1]
    x_s = std_scaler.fit_transform(x)

    # Plots

    # Box Plots
    for i in range(0, 4):
        plot = px.box(
            data_frame=df,
            y=col_names[i],
            x="class",
            color="class",
            points="all",
            title="Box Plot of {}".format(col_names[i]),
        )
        plot.write_html(
            file="./Plots/Box plot of {}.html".format(col_names[i]),
            include_plotlyjs="cdn",
        )

    # Violin Plots
    for i in range(0, 4):
        plot = px.violin(
            data_frame=df,
            y=col_names[i],
            x="class",
            color="class",
            title="Violin Plot of {}".format(col_names[i]),
        )
        plot.write_html(
            file="./Plots/Violin plot of {}.html".format(col_names[i]),
            include_plotlyjs="cdn",
        )

    # Grouped Box Plot
    # Source: https://statisticsglobe.com/draw-plotly-boxplot-python
    df_grouped = df.set_index("class").stack().reset_index()
    df_grouped.columns = ["class", "attribute", "value"]
    print(df_grouped.head())

    fig_grouped_box = px.box(
        data_frame=df_grouped,
        y="value",
        x="attribute",
        color="class",
        points="all",
        title="Grouped Box Plot of Iris Species",
    )

    fig_grouped_box.write_html(
        file="./Plots/Grouped Box Plot of Iris Species.html", include_plotlyjs="cdn"
    )

    # Grouped Violin Plot
    fig_grouped_violin = px.violin(
        data_frame=df_grouped,
        y="value",
        x="attribute",
        color="class",
        title="Grouped Violin Plot of Iris Species",
    )

    fig_grouped_violin.write_html(
        file="./Plots/Grouped Violin Plot of Iris Species.html", include_plotlyjs="cdn"
    )

    # Scatter Matrix Plot
    scatter_matrix = px.scatter_matrix(
        df,
        dimensions=["sepal length", "sepal width", "petal length", "petal width"],
        color="class",
        symbol="class",
        title="Scatter Matrix of Iris Data Set",
    )

    scatter_matrix.write_html(
        file="./Plots/Scatter Matrix of Iris Species.html", include_plotlyjs="cdn"
    )

    # Grouped Bar Plot
    fig_grouped_bar = px.bar(
        data_frame=df_grouped,
        y="value",
        x="attribute",
        color="class",
        title="Grouped Bar Plot of Iris Species",
        barmode="group",
    )

    fig_grouped_bar.write_html(
        file="./Plots/Grouped Bar Plot of Iris Species.html", include_plotlyjs="cdn"
    )

    # Label Encoder and encode class labels in Iris dataset
    # Source: https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/
    label_encoder = LabelEncoder()
    y = df["class"]
    y = label_encoder.fit_transform(y)

    # Train and Test Split
    x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(
        x_s, y, test_size=0.3, random_state=0
    )

    # Fit the training data to random forest classifier
    rf_classifier = RandomForestClassifier(random_state=0)
    rf_classifier.fit(x_s_train, y_s_train)

    # Predict iris class using test data
    y_predicted = rf_classifier.predict(x_s_test)

    # Print the classification report of random forest classifier
    print(
        "\n The classification report of Random Forest Classifier(not included in pipeline) is \n {} \n".format(
            classification_report(y_s_test, y_predicted)
        )
    )

    # Now, let's build a pipeline of the classifiers
    pipeline_dictionary = {
        "Support_Vectors_Classifier": SVC(),
        "KNN_Classifier": KNeighborsClassifier(),
        "Decision_Tree_Classifier": DecisionTreeClassifier(),
        "Random_Forest_Classifier": RandomForestClassifier(),
    }

    # Train and Test Split for the pipeline
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )

    # Building Models using pipeline
    model_fit_test(
        classifier_dict=pipeline_dictionary,
        train_var=x_train,
        train_target=y_train,
        test_var=x_test,
        test_target=y_test,
    )

    # Create Boolean responses for classes
    df = pd.get_dummies(df)
    print(df.columns)

    class_iris_setosa_mean = df[df["class_Iris-setosa"] == 1]["sepal length"].mean()
    class_not_iris_setosa_mean = df[df["class_Iris-setosa"] == 0]["sepal length"].mean()

    total_mean = df["sepal length"].mean()

    fig = go.Figure(layout=dict(bargap=0.3))

    fig.add_trace(
        go.Histogram(
            x=df["class_Iris-setosa"],
            y=df["sepal length"],
            name="Count",
            yaxis="y",
            opacity=0.5,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[class_not_iris_setosa_mean, class_iris_setosa_mean],
            name="Bin Mean",
            yaxis="y2",
        )
    )

    fig.add_trace(
        go.Scatter(x=[0, 1], y=[total_mean, total_mean], name="Total Mean", yaxis="y2")
    )

    # axes objects
    fig.update_layout(
        xaxis=dict(domain=[0.3, 0.7], title="Class Iris Setosa"),
        # 1st y axis
        yaxis=dict(
            title="Count",
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
        ),
        # 2nd y axis
        yaxis2=dict(title="Sepal Length", overlaying="y", side="right"),
    )

    # title
    fig.update_layout(
        title_text="Difference with Mean of Response Plot on Data",
        width=800,
    )

    fig.write_html(file="./Plots/Diff Plot Iris Setosa.html", include_plotlyjs="cdn")

    return


if __name__ == "__main__":
    sys.exit(main())
