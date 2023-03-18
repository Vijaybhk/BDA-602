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
        input_df: DataFrame = None,
        predictors: list[str] = None,
        response: str = None,
    ):
        """
        Constructor for the Variable Processor class
        :param input_df: Input Pandas dataframe
        :param predictors: list of predictors
        :param response: Response Variable Name
        """
        self.input_df = input_df
        self.predictors = predictors
        self.response = response

    def check_continuous_var(self, var: str) -> bool:
        """
        Method to check if the variable is continuous or not
        :param var: Input Variable Name
        :return: True if Continuous else False
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

    def get_cat_and_cont_predictors(self) -> tuple[list, list]:
        """
        Method to get the lists of Categorical and Continuous Predictors
        """
        cat_predictors = []
        cont_predictors = []

        for predictor in self.predictors:
            if self.check_continuous_var(var=predictor):
                cont_predictors.append(predictor)
            else:
                cat_predictors.append(predictor)

        return cat_predictors, cont_predictors

    def get_regression_scores(self, cont_predictors: list[str]) -> tuple[dict, dict]:
        """
        Method to execute logistic or linear regression for each variable
        based on response type and get the p values and t scores.
        :param cont_predictors: list of continuous predictors
        :return: dictionaries of t scores and p values with predictor name as key
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
            # print(f"Variable: {column}")
            # print(linear_regression_model_fitted.summary())

            t_dict[column] = round(regression_model_fitted.tvalues[1], 6)
            p_dict[column] = "{:.6e}".format(regression_model_fitted.pvalues[1])

        return t_dict, p_dict

    def get_random_forest_scores(self, cont_predictors: list[str]) -> dict:
        """
        Method to execute Random Forest Classifier or Regressor based on
        response type and get the feature importance scores.
        :param cont_predictors: list of continuous predictors
        :return: dictionary of variable importance scores with predictor name as key
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
