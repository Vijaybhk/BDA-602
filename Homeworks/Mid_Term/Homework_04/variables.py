from __future__ import annotations

import sys

import pandas.api.types as pt
import statsmodels.api as sm
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class DfVariableProcessor:
    """
    Variable Processor class to process independent
    and dependent variables in the dataframe
    """

    def __init__(
        self,
        input_df: DataFrame | None = None,
        predictors: list[str] | None = None,
        response: str | None = None,
    ):
        """
        Constructor for the Variable Processor class

        Parameters
        -------------

            input_df: DataFrame
                Input Pandas dataframe
            predictors: list[str]
                list of predictors
            response: str
                Response Variable Name

        """
        self.input_df = input_df
        self.predictors = predictors
        self.response = response

    def check_continuous_var(self, var: str) -> bool:
        """
        Method to check if the variable is continuous or not

        Parameters
        -------------
            var: str
                Input Variable Name

        Returns
        -------------
            flag_cont: bool
                True if Continuous else False

        """
        flag_cont = False

        column = self.input_df[var]

        if pt.is_float_dtype(column):
            flag_cont = True

        elif pt.is_integer_dtype(column) and tuple(column.unique()) not in [
            (1, 0),
            (0, 1),
        ]:
            flag_cont = True

        return flag_cont

    def get_response_type(self) -> str:
        """
        Method to get the response type. Converts bool(True/False) to 1 and 0 to
        support remaining code. Responses other than boolean categorical or
        continuous will be returned unsupported.

        Returns
        -------------
        str
            Outputs Response Type
            "Categorical", or "Continuous", or "Unsupported Categorical"

        """

        res_column = self.input_df[self.response]

        if pt.is_bool_dtype(res_column):
            self.input_df[self.response] = self.input_df[self.response].astype(int)
            return "categorical"

        elif pt.is_integer_dtype(res_column) and tuple(res_column.unique()) in [
            (1, 0),
            (0, 1),
        ]:
            return "categorical"

        elif self.check_continuous_var(var=self.response):
            return "continuous"

        else:
            print("Unsupported Categorical Response Type")
            return "Unsupported Categorical"

    def get_cat_and_cont_predictors(self) -> tuple[list, list, dict]:
        """
        Method to get the lists of Categorical and Continuous Predictors,
        and dictionary of predictor types

        Returns
        -------------
            cat_predictors : list
                lists of Categorical Predictors
            cont_predictors : list
                lists of Continuous Predictors
            pred_type_dict : dict
                dictionary of predictor types

        """
        cat_predictors = []
        cont_predictors = []
        pred_type_dict = {}

        for predictor in self.predictors:
            if self.check_continuous_var(var=predictor):
                cont_predictors.append(predictor)
                pred_type_dict[predictor] = "Continuous"
            else:
                cat_predictors.append(predictor)
                pred_type_dict[predictor] = "Categorical"

        return cat_predictors, cont_predictors, pred_type_dict

    def get_regression_scores(self, cont_predictors: list[str]) -> tuple[dict, dict]:
        """
        Method to execute logistic or linear regression for each variable
        based on response type and get the p values and t scores.

        Parameters
        -------------
            cont_predictors: list[str]
                list of continuous predictors

        Returns
        -------------
            t_dict: dict
                dictionaries of t scores with predictor name as key
            p_dict: dict
                dictionaries of p values with predictor name as key

        """
        t_dict = {}
        p_dict = {}

        res_type = self.get_response_type()
        regression_model = None

        for column in cont_predictors:
            x = self.input_df[column]
            y = self.input_df[self.response]
            predictor = sm.add_constant(x)

            if res_type == "continuous":
                regression_model = sm.OLS(y, predictor)
            elif res_type == "categorical":
                regression_model = sm.Logit(y, predictor)

            regression_model_fitted = regression_model.fit(disp=False)

            t_dict[column] = round(regression_model_fitted.tvalues[1], 6)
            p_dict[column] = "{:.6e}".format(regression_model_fitted.pvalues[1])

        return t_dict, p_dict

    def get_random_forest_scores(self, cont_predictors: list[str]) -> dict:
        """
        Method to execute Random Forest Classifier or Regressor based on
        response type and get the feature importance scores.

        Parameters
        -------------
            cont_predictors: list[str]
                list of continuous predictors

        Returns
        -------------
            scores_dict: dict
                dictionaries of Random Forest variable importance scores
                with predictor name as key

        """

        x = self.input_df[cont_predictors]
        y = self.input_df[self.response]
        rf_model = None

        res_type = self.get_response_type()

        if res_type == "continuous":
            rf_model = RandomForestRegressor(random_state=0)
            rf_model.fit(x, y)

        elif res_type == "categorical":
            rf_model = RandomForestClassifier(random_state=0)
            rf_model.fit(x, y)

        scores = rf_model.feature_importances_
        scores_dict = {}

        for index, predictor in enumerate(cont_predictors):
            scores_dict[predictor] = scores[index]

        return scores_dict


def main():
    help(DfVariableProcessor)


if __name__ == "__main__":
    sys.exit(main())
