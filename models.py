from pyforest import *
from abc import abstractmethod, ABC
import hashlib

__all__ = [
    'LinearRegression',
    'kNearestNeighborsClassifier'
]

class ModelProfile:
    def __init__(self, target_name, predictor_name, model_name,
                 hyperparameters = None, scores = None, model=None):
        """
        Model Profile stores the properties of a model.
        :param target_name:
        :param predictor_name: List of strings, denoting column names
        :param model_name: Linear_regression, kNN, etc. Can be changed in future.
        :param hyperparameters:
        :param scores:
        :param model:
        """
        self.target_name = target_name
        self.predictor_name = predictor_name
        self.model_name = model_name
        self.hyperparameteres = hyperparameters
        self.scores = scores
        self.model = model
        # self.file_name = hashlib.sha1(target_name + ','.join(predictor_name) + model_name + \
        #                                   hyperparameters).hexdigest() + ".pkl"

class ModelNotFoundError(Exception):
    # Raised when the model for a particular target_col is not found
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Models(ABC):

    def __init__(self):
        """
        self.model_profile_mapping = mapping from target_column names to list of ModelProfile
        """
        self.model_profile_mapping = Models.__load_model_profiles()

    @staticmethod
    def __load_model_profiles():
        try:
            with open(r"ml_models/model_profile_mapping.pkl", "rb") as f:
                return pickle.load(f)
        except(FileNotFoundError):
            print("Making a new model_profile_mapping from scratch")
            return {}

    def show_model_profiles(self, target_col, pretty_print = True):
        if target_col not in self.model_profile_mapping.keys():
            raise ModelNotFoundError("Model for the given target_col not present")

        for model_profile in self.model_profile_mapping[target_col]:
            if not pretty_print:
                print(model_profile.__dict__)
            else:
                for key in model_profile.__dict__:
                    print(key, " : ", model_profile.__dict__[key])
            print()

    def predict(self, df, target_col, profile_no = 0):
        if target_col not in self.model_profile_mapping:
            raise ModelNotFoundError("Model for the given target_col not present")

        if profile_no > len(self.model_profile_mapping[target_col])-1:
            raise ModelNotFoundError("profile_no for the given target_col not present")

        predictor_col = self.model_profile_mapping[target_col][profile_no].predictor_name
        model = self.model_profile_mapping[target_col][profile_no].model
        return model.predict((df.loc[:, predictor_col]).
                             dropna(axis=0, how='any', inplace=False))

    @abstractmethod
    def create_model(self, df, target_col, predictor_col, *args, **kwargs):
        pass

    def delete_model_profile(self, target_col, profile_no = 0):
        if target_col not in self.model_profile_mapping.keys():
            raise ModelNotFoundError("Model for the given target_col not present")

        del self.model_profile_mapping[target_col][profile_no]
        if len(self.model_profile_mapping[target_col]) == 0:
            del self.model_profile_mapping[target_col]

        self.save_model_profile()
        print("Successfully deleted")

    def save_model_profile(self):
        with open(r"ml_models/model_profile_mapping.pkl", "wb") as f:
            pickle.dump(self.model_profile_mapping, f)

    def create_model_profile(self, target_name, predictor_name, model_name, hyperparameters,
                           scores, model):
        profile = ModelProfile(target_name, predictor_name, model_name, hyperparameters,
                               scores, model)
        if target_name not in self.model_profile_mapping.keys():
            self.model_profile_mapping[target_name] = [profile]
        else:
            self.model_profile_mapping[target_name].append(profile)

        self.save_model_profile()

    def fill_missing(self, df, target_col, profile_no=0, inplace=False):
        if target_col not in self.model_profile_mapping:
            raise ModelNotFoundError("Model for the given target_col not present")

        if profile_no > len(self.model_profile_mapping[target_col]) - 1:
            raise ModelNotFoundError("profile_no for the given target_col not present")

        query_str = target_col + '!=' + target_col
        if df.query(query_str).shape[0] == 0:
            print("No values missing from target_col")
            if inplace:
                return
            else:
                return df

        predictor_col = self.model_profile_mapping[target_col][profile_no].predictor_name
        predictor_index = df.loc[:, predictor_col]. \
            dropna(axis=0, how='any', inplace=False).index
        target_index = df.loc[df.loc[:, target_col].isna()].index
        fill_index = set(target_index).intersection(set(predictor_index))

        if len(fill_index) == 0:
            print("No values missing from target_col where predictor_col is not null")
            if inplace:
                return
            else:
                return df

        model = self.model_profile_mapping[target_col][profile_no].model
        fill_values = model.predict(df.loc[fill_index, predictor_col])

        if inplace:
            df.loc[:, target_col].fillna(pd.Series(fill_values, index=fill_index),
                                         inplace=inplace)
        else:
            temp_df = df.copy(deep=True)
            temp_df.loc[:, target_col].fillna(pd.Series(fill_values, index=fill_index),
                                              inplace=True)
            return temp_df

class LinearRegression(Models):

    def __init__(self):
        super().__init__()
        self.train = 0.8

    def create_model(self, df, target_col, predictor_col, *args, **kwargs):
        """

        :param df:
        :param target_col:
        :param predictor_col:
        :param args:
        :param kwargs:
        :return:
        """
        temp_df = df.loc[:, [target_col]+predictor_col].\
            dropna(axis = 0, how = 'any', inplace = False)
        train_index = temp_df.sample(frac = self.train, replace = False, axis = 0).index
        test_index = temp_df.iloc[~temp_df.index.isin(train_index)].index

        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(temp_df.loc[train_index, predictor_col],
                                   temp_df.loc[train_index, target_col])

        from sklearn.metrics import r2_score
        score = r2_score(temp_df.loc[test_index, target_col],
                        reg.predict(temp_df.loc[test_index, predictor_col]))

        self.create_model_profile(target_col, predictor_col, "LinearRegression", {},
                                {"r2_score":score}, reg)


class kNearestNeighborsClassifier(Models):

    def __init__(self):
        super().__init__()
        self.train = 0.8

    def create_model(self, df, target_col, predictor_col, *args, **kwargs):
        """

        :param df:
        :param target_col:
        :param predictor_col:
        :param args:
        :param kwargs:
        :return:
        """
        temp_df = df.loc[:, [target_col]+predictor_col].\
            dropna(axis = 0, how = 'any', inplace = False)
        train_index = temp_df.sample(frac = self.train, replace = False, axis = 0).index
        test_index = temp_df.iloc[~temp_df.index.isin(train_index)].index

        from sklearn.neighbors import KNeighborsClassifier
        # Do responsibly pass on all the required hyper-parameters
        neigh = KNeighborsClassifier(**kwargs)
        neigh.fit(temp_df.loc[train_index, predictor_col],
                                   temp_df.loc[train_index, target_col])

        pred = neigh.predict(temp_df.loc[test_index, predictor_col])

        from sklearn.metrics import matthews_corrcoef, zero_one_loss, f1_score

        matthews_corrcoef_score = matthews_corrcoef(temp_df.loc[test_index, target_col],
                                                    pred)
        zero_one_loss_score = zero_one_loss(temp_df.loc[test_index, target_col],
                                            pred)
        f1_score_score_micro = f1_score(temp_df.loc[test_index, target_col],
                                        pred, average="micro")
        f1_score_score_macro = f1_score(temp_df.loc[test_index, target_col],
                                        pred, average="macro")
        f1_score_score_weighted = f1_score(temp_df.loc[test_index, target_col],
                                           pred, average="weighted")

        scores = {"matthews_corrcoef_score" : matthews_corrcoef_score,
                  "zero_one_loss_score" : zero_one_loss_score,
                  "f1_score_score_micro" : f1_score_score_micro,
                  "f1_score_score_macro" : f1_score_score_macro,
                  "f1_score_score_weighted" : f1_score_score_weighted}

        self.create_model_profile(target_col, predictor_col, "kNearestNeighborsClassifier",
                                  kwargs, scores = scores, model = neigh)
