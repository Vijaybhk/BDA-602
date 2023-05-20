from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
from generator import generate_report
from plotter import combine_html
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from utilities import mariadb_df, model_results


def main():
    data_name = "Baseball"
    query = "SELECT" + " * FROM features_table"
    df = mariadb_df(query, db_host="vij_mariadb:3306")
    df.sort_values(by=["game_date", "game_id"], inplace=True, ignore_index=True)
    df.set_index("game_id", inplace=True)
    train_size = int(len(df) * 0.6)
    train = df.iloc[:train_size, 1:]
    test = df.iloc[train_size:, 1:]
    cols = train.columns
    medians = train.median(axis=0, skipna=True)

    # Handling Nulls
    # Replacing nulls in training set and testing set with medians of training set
    for col_idx, col in enumerate(cols):
        train[col] = train[col].fillna(medians[col_idx])
        test[col] = test[col].fillna(medians[col_idx])

    df = pd.concat([train, test])

    # Getting predictors and response from dataframe
    predictors = df.columns[:-1]
    response = df.columns[-1]

    # Plots directory path
    this_dir = os.path.dirname(os.path.realpath(__file__))
    plot_dir = f"{this_dir}/Output/Plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Creating Predictors report as html
    generate_report(df, predictors, response, plot_dir, data_name)

    features = [
        "SP_BFP_DIFF_ROLL",
        "SP_FIP_DIFF_HIST",
        "SP_SO9_DIFF_ROLL",
        "SP_SOPP_DIFF_HIST",
        "TB_BABIP_DIFF_ROLL",
        "TB_OPS_DIFF_HIST",
        "TM_PYEX_DIFF_HIST",
        "TM_RD_DIFF_ROLL",
    ]

    x = df[features]

    x_train = x.iloc[:train_size, :].values
    x_test = x.iloc[train_size:, :].values
    y_train = df.iloc[:train_size, -1].values
    y_test = df.iloc[train_size:, -1].values

    # Logistic Regression

    logr_pipe = Pipeline(
        [("std_scaler", StandardScaler()), ("classifier", LogisticRegression())]
    )
    logr_pipe = logr_pipe.fit(x_train, y_train)
    logr_res = model_results("LogisticReg", logr_pipe, x_test, y_test, plot_dir)

    # Decision Tree Classifier
    tree_random_state = 42
    decision_tree = DecisionTreeClassifier(random_state=tree_random_state)
    decision_tree.fit(x_train, y_train)
    dtree_res = model_results("DecisionTree", decision_tree, x_test, y_test, plot_dir)

    # Bagging Tree Classifier

    bagging_tree = BaggingClassifier(
        estimator=decision_tree, random_state=tree_random_state
    )
    bagging_tree.fit(x_train, y_train)
    bag_res = model_results("BaggingTree", bagging_tree, x_test, y_test, plot_dir)

    # Random Forest Model

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(x_train, y_train)
    rf_res = model_results("RandomForest", rf_model, x_test, y_test, plot_dir)

    # AdaBoost Classifier

    ada = AdaBoostClassifier(random_state=42)
    ada.fit(x_train, y_train)
    ada_res = model_results("AdaBoost", ada, x_test, y_test, plot_dir)

    # XG Boost Classifier

    xg = GradientBoostingClassifier(random_state=42)
    xg.fit(x_train, y_train)
    xg_res = model_results("XGBoost", xg, x_test, y_test, plot_dir)

    # KNN Classifier

    knn = Pipeline(
        [("std_scaler", StandardScaler()), ("classifier", KNeighborsClassifier())]
    )
    knn.fit(x_train, y_train)
    knn_res = model_results("KNN", knn, x_test, y_test, plot_dir)

    # Calculate predicted probabilities
    logr_prob = logr_pipe.predict_proba(x_test)[::, 1]
    rf_prob = rf_model.predict_proba(x_test)[::, 1]
    dtree_prob = decision_tree.predict_proba(x_test)[::, 1]
    bag_prob = bagging_tree.predict_proba(x_test)[::, 1]
    ada_prob = ada.predict_proba(x_test)[::, 1]
    xg_prob = xg.predict_proba(x_test)[::, 1]
    knn_prob = knn.predict_proba(x_test)[::, 1]

    # Calculate the ROC curve points
    logr_fpr, logr_tpr, _ = roc_curve(y_test, logr_prob)
    dtree_fpr, dtree_tpr, _ = roc_curve(y_test, dtree_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    bag_fpr, bag_tpr, _ = roc_curve(y_test, bag_prob)
    ada_fpr, ada_tpr, _ = roc_curve(y_test, ada_prob)
    xg_fpr, xg_tpr, _ = roc_curve(y_test, xg_prob)
    knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_prob)

    # Calculate the AUC
    logr_auc = logr_res[2]
    dtree_auc = dtree_res[2]
    rf_auc = rf_res[2]
    bag_auc = bag_res[2]
    ada_auc = ada_res[2]
    xg_auc = xg_res[2]
    knn_auc = knn_res[2]

    # Create the figure
    fig = go.Figure(
        data=go.Scatter(
            x=logr_fpr,
            y=logr_tpr,
            name=f"Logistic Regression (AUC={round(logr_auc, 6)})",
        )
    )
    fig = fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            line=dict(dash="dash"),
            mode="lines",
            showlegend=False,
        )
    )

    fig = fig.add_trace(
        go.Scatter(x=rf_fpr, y=rf_tpr, name=f"Random Forest (AUC={round(rf_auc, 6)})")
    )
    fig = fig.add_trace(
        go.Scatter(
            x=dtree_fpr, y=dtree_tpr, name=f"Decision Tree (AUC={round(dtree_auc, 6)})"
        )
    )
    fig = fig.add_trace(
        go.Scatter(x=bag_fpr, y=bag_tpr, name=f"Bagging Tree (AUC={round(bag_auc, 6)})")
    )
    fig = fig.add_trace(
        go.Scatter(x=ada_fpr, y=ada_tpr, name=f"Ada Boost (AUC={round(ada_auc, 6)})")
    )
    fig = fig.add_trace(
        go.Scatter(x=xg_fpr, y=xg_tpr, name=f"XG Boost (AUC={round(xg_auc, 6)})")
    )
    fig = fig.add_trace(
        go.Scatter(
            x=knn_fpr, y=knn_tpr, name=f"KNN Classifier (AUC={round(knn_auc, 6)})"
        )
    )

    # Label the figure
    fig.update_layout(
        title="Receiver Operator Characteristic ROC Curve",
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
    )

    fig.write_html(file=f"{plot_dir}/roc.html", include_plotlyjs="cdn")

    # Table of Model Results
    df_list = [logr_res, rf_res, dtree_res, bag_res, ada_res, xg_res, knn_res]

    results = pd.DataFrame(
        df_list,
        columns=[
            "Model",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "f1-Score",
            "MeanAbsError",
            "MatthewsCorrCoeff",
        ],
    )

    results.sort_values(
        by=["Accuracy", "Recall"], ascending=[False, False], inplace=True
    )
    results.to_html(f"{plot_dir}/results.html", escape=False, index=False)

    # Models Results as html
    combine_html(
        combines={
            "<h3> Summary of Model Results <h3>": f"{plot_dir}/results.html",
            "<h2> Receiver Operator Characteristic (ROC) Curves </h2>": f"{plot_dir}/roc.html",
            "<h2> Logistic Regression <h2>": f"{plot_dir}/LogisticReg_cm.html",
            "<h3> LR Classification Report <h3>": f"{plot_dir}/LogisticReg_cr.html",
            "<h2> Decision Tree <h2>": f"{plot_dir}/DecisionTree_cm.html",
            "<h3> DTree Classification Report <h3>": f"{plot_dir}/DecisionTree_cr.html",
            "<h2> Bagging Tree <h2>": f"{plot_dir}/BaggingTree_cm.html",
            "<h3> BTree Classification Report <h3>": f"{plot_dir}/BaggingTree_cr.html",
            "<h2> Random Forest <h2>": f"{plot_dir}/RandomForest_cm.html",
            "<h3> RF Classification Report <h3>": f"{plot_dir}/RandomForest_cr.html",
            "<h2> Ada Boost <h2>": f"{plot_dir}/AdaBoost_cm.html",
            "<h3> Ada Boost Classification Report <h3>": f"{plot_dir}/AdaBoost_cr.html",
            "<h2> XG Boost <h2>": f"{plot_dir}/XGBoost_cm.html",
            "<h3> XG Classification Report <h3>": f"{plot_dir}/XGBoost_cr.html",
            "<h2> KNN <h2>": f"{plot_dir}/KNN_cm.html",
            "<h3> KNN Classification Report <h3>": f"{plot_dir}/KNN_cr.html",
        },
        result="Output/Model_Results.html",
        head=f"{data_name} Models Report",
    )

    return


if __name__ == "__main__":
    sys.exit(main())
