import pandas as pd


class DfVariableProcessor:
    def __init__(self, input_df=None, predictors=None, response=None):
        self.input_df = input_df
        self.predictors = predictors
        self.response = response

    def check_categorical(self, var):
        flag_cat = False

        if pd.api.types.is_string_dtype(self.input_df[var]):
            flag_cat = True

        elif pd.api.types.is_bool_dtype(self.input_df[var]):
            flag_cat = True

        elif pd.api.types.is_integer_dtype(self.input_df[var]) and tuple(
            self.input_df[var].unique()
        ) in [(1, 0), (0, 1)]:
            flag_cat = True

        return "categorical" if flag_cat is True else "continuous"

    def get_response_type(self):
        response_type = self.check_categorical(var=self.response)
        return response_type

    def get_cat_and_cont_predictors(self):
        cat_predictors = []
        cont_predictors = []

        for predictor in self.predictors:
            ptype = self.check_categorical(var=predictor)
            if ptype == "categorical":
                cat_predictors.append(predictor)
            elif ptype == "continuous":
                cont_predictors.append(predictor)

        return cat_predictors, cont_predictors
