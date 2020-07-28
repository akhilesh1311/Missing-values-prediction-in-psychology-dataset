import numpy as np
import pandas as pd
import string

__all__ = [
    'Preprocessors'
]

class PreprocessorOrderError(Exception):
    """Raised when the order of preprocessing is not correct."""
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Preprocessors:
    def __init__(self):
        """
        This variable is used for performing sanity checks if a Preprocessor has been run before.
        Useful for some Preprocessors.
        DoneTODO: Decide if this variable should be an instance variable or a class variable
        """
        self.__func_hist = {"put_nans": False,
                   "reduce_classes": False,
                   "reduce_classes_race" : False,
                   "make_ordinal_scales": False,
                   "make_iat_sensible_ordinal": False,
                   "clean_exprunafter2": False,
                   "make_flagsupplement_ordinal": False,
                   "make_sysjust_ordinal": False,
                   "translate_text": False,
                   "preprocess_partgender" : False,
                   "add_midnight" : False,
                    "remove_single_value_cols" : False,
                            "create_power_set" : False,
                            "fit_transform_label_encoder" : False}

        self.cols = {'regression_cols' : [],
                     'nlp_cols' : [],
                     'remove_cols' : [],
                     'ordinal_regression_cols' : [],
                     'classification_cols' : [],
                     'single_value_cols' : [],
                     'power_set_removed_rows' : [],
                     'power_set_removed_cols' : []
                     }

        self.label_encoders = {}

    def reset(self):
        """
        Erases history of all the run preprocessors.
        Useful if we are dealing with a new dataframe.
        :return:
        """
        self.__func_hist = {key : False for key in self.__func_hist.keys()}

    def reset_all(self):
        """
        Erases history of all the run preprocessors, along with all the cols.
        Useful if we are dealing with a new dataframe.
        :return:
        """
        self.__func_hist = {key : False for key in self.__func_hist.keys()}
        self.cols = {key : [] for key in self.cols.keys()}

    def put_nans(self, df, inplace = False):
        '''
        Converts all strings to lowercase.
        Checks for any useless values like '.', '#', ' ', 'no response' with np.NaN.
        Answer is stripped off whitespaces.
        By default, it is not done inplace.
        '''
        self.__func_hist["put_nans"] = True
        def check_replace(s):
            if type(s) != str:
                return s
            s = s.lower()
            regex = string.punctuation + ' ' + 'null'
            all_special = all(i in regex for i in s)
            if all_special or s == 'no response' or s == 'not selected' or \
                s == 'debriefing':
                return np.NaN
            else:
                return s.strip()

        if inplace:
            df.iloc[:, :] = df.applymap(check_replace)
            return
        else:
            return df.applymap(check_replace)

    def reduce_classes(self, df, col, no_classes=10, inplace = False, other_class='other'):
        '''
        Reduces the number of features in 'col' variable. Might or might not be useful.
        Assumes that put_nans function is run before running this.

        Specific reduce_classes functions might be more useful.

        This function chooses the top 'no_classes-1' classes as valid classes, and groups other classes
        into 'other_class' variable.

        no_classes > 0
        '''
        if not self.__func_hist["put_nans"]:
            raise PreprocessorOrderError("Run put_nans function first")
        self.__func_hist["reduce_classes"] = True

        indices = df.loc[:, col].value_counts().iloc[:no_classes - 1].index
        if inplace:
            df.loc[:, col] = df.loc[:, col].apply(lambda x: x if pd.isnull(x) else \
                (other_class if x not in indices else x))
            return
        else:
            temp_df = df.copy(deep=True)
            temp_df.loc[:, col] = temp_df.loc[:, col].apply(lambda x: x if pd.isnull(x) else \
                (other_class if x not in indices else x))
            return temp_df

    def reduce_classes_race(self, df, inplace = False):
        '''
        Specific function for race variable. Might be useful since machine can't decode semantic differences.
        Reduces the number of features in race variable. Might or might not be useful.
        Assumes that put_nans function is run before running this.
        By default, it is done inplace.
        '''
        if not self.__func_hist["put_nans"]:
            raise PreprocessorOrderError("Run put_nans function first")
        self.__func_hist["reduce_classes_race"] = True

        race_dict = {'White': 'White',
                     'Black or African American': 'Black or African American',
                     'East Asian': 'East Asian',
                     'Other or Unknown': 'Other or Unknown',
                     'South Asian': 'South Asian',
                     'More than one race - Other': 'More than one race - Other',
                     'turk': 'turk',
                     'chinese': 'East Asian',
                     'Nederlands': 'Other or Unknown',
                     'More than one race - Black/White': 'More than one race - Other',
                     'brazilwhite': 'Other or Unknown',
                     'brazilbrown': 'Other or Unknown',
                     'American Indian/Alaska Native': 'Other or Unknown',
                     'Native Hawaiian or other Pacific Islander': 'Other or Unknown',
                     'brazilblack': 'Other or Unknown',
                     'brazilyellow': 'Other or Unknown',
                     'indian': 'South Asian',
                     'malay': 'Other or Unknown',
                     'Nederlandse': 'Other or Unknown',
                     'nederlands': 'Other or Unknown',
                     'Belgisch Nederlands': 'Other or Unknown',
                     'Marokkaans Nederlands': 'Other or Unknown',
                     'brazilindigenous': 'Other or Unknown',
                     'italiaans nederlands': 'Other or Unknown',
                     'Turks Nederlands': 'Other or Unknown',
                     'duits': 'Other or Unknown',
                     'Russian': 'Other or Unknown',
                     'nl': 'Other or Unknown',
                     'Duits': 'Other or Unknown'}
        race_dict = {key.lower(): val.lower() for key, val in race_dict.items()}

        if inplace:
            df.race = df.race.apply(lambda x: x if pd.isnull(x) else race_dict[x])
            return
        else:
            temp_df = df.copy(deep=True)
            temp_df.race = temp_df.race.apply(lambda x: x if pd.isnull(x) else race_dict[x])
            return temp_df

    def make_ordinal_scales(self, df, inplace = False):
        """
        Makes scalesa and scalesb ordinal by appending integers in front of the string.
        This way, we get the desired encoding on passing these columns to LabelEncoder.
        ..deprecated - use sklearn.preprocessing.OrdinalEncoder instead
        :param df: dataframe containing scalesa and scalesb variables
        :param inplace:
        :return: modified dataframe if inplace == True
        """

        def append_number_a(x):
            if pd.isnull(x):
                return x
            elif x == 'Up to a half hour'.lower():
                return '0_' + x
            elif x == 'Half an hour to an hour'.lower():
                return '1_' + x
            elif x == 'One to one and a half hours'.lower():
                return '2_' + x
            elif x == 'One and a half to two hours'.lower():
                return '3_' + x
            elif x == 'Two to two and a half hours'.lower():
                return '4_' + x
            elif x == 'More than two and a half hours'.lower():
                return '5_' + x
            else:
                return x

        def append_number_b(x):
            if pd.isnull(x):
                return x
            elif x == 'Up to two and a half hours'.lower():
                return '0_' + x
            elif x == 'Two and a half to three hours'.lower():
                return '1_' + x
            elif x == 'Three to three and a half hours'.lower():
                return '2_' + x
            elif x == 'Three and a half to four hours'.lower():
                return '3_' + x
            elif x == 'Four to four and a half hours'.lower():
                return '4_' + x
            elif x == 'More than four and a half hours'.lower():
                return '5_' + x
            else:
                return x

        if inplace:
            df.scalesa = df.scalesa.apply(append_number_a)
            df.scalesb = df.scalesb.apply(append_number_b)
            return
        else:
            temp_df = df.copy(deep=True)
            temp_df.scalesa = temp_df.scalesa.apply(append_number_a)
            temp_df.scalesb = temp_df.scalesb.apply(append_number_b)
            return temp_df

    def make_iat_sensible_ordinal(self, df, inplace = False):
        '''
        Converts the compressed form of values in iatexplicit experiments to human readable format.
        Scale is from 1-7, with art missing the scale value of '3' and math missing the scale value of '7'.
        Also, makes iatexplicit variables ordinal for LabelEncoder.
        ..deprecated - use sklearn.preprocessing.OrdinalEncoder instead
        TODO: How to predict '3' in art and '7' in math?
        '''
        iatexplicitart_dict = [0 for _ in range(7)]
        iatexplicitart_dict[1] = {'very bad': '1_very bad',
                                  'moderately bad': '2_moderately bad',
                                  'slightly bad': '3_slightly bad',
                                  '4': '4_neither good nor bad',
                                  '5': '5_slightly good',
                                  '6': '6_moderately good',
                                  '7': '7_very good'}
        iatexplicitart_dict[2] = {'very sad': '1_very sad',
                                  'moderately sad': '2_moderately sad',
                                  'slightly sad': '3_slightly sad',
                                  '4': '4_neither happy nor sad',
                                  '5': '5_slightly happy',
                                  '6': '6_moderately happy',
                                  '7': '7_very happy'}
        iatexplicitart_dict[3] = {'very ugly': '1_very ugly',
                                  'moderately ugly': '2_moderately ugly',
                                  'slightly ugly': '3_slightly ugly',
                                  '4': '4_neither beautiful nor ugly',
                                  '5': '5_slightly beautiful',
                                  '6': '6_moderately beautiful',
                                  '7': '7_very beautiful'}
        iatexplicitart_dict[4] = {'very disgusting': '1_very disgusting',
                                  'moderately disgusting': '2_moderately disgusting',
                                  'slightly disgusting': '3_slightly disgusting',
                                  '4': '4_neither delightful nor disgusting',
                                  '5': '5_slightly delightful',
                                  '6': '6_moderately delightful',
                                  '7': '7_very delightful'}
        iatexplicitart_dict[5] = {'very avoid': '1_very avoid',
                                  'moderately avoid': '2_moderately avoid',
                                  'slightly avoid': '3_slightly avoid',
                                  '4': '4_neither approach nor avoid',
                                  '5': '5_slightly approach',
                                  '6': '6_moderately approach',
                                  '7': '7_very approach'}
        iatexplicitart_dict[6] = {'very afraid': '1_very afraid',
                                  'moderately afraid': '2_moderately afraid',
                                  'slightly afraid': '3_slightly afraid',
                                  '4': '4_neither unafraid nor afraid',
                                  '5': '5_slightly unafraid',
                                  '6': '6_moderately unafraid',
                                  '7': '7_very unafraid'}
        if inplace:
            for i in range(1, 7):
                df.loc[:, 'iatexplicitart' + str(i)] = df.loc[:, 'iatexplicitart' + str(i)].map(iatexplicitart_dict[i])
                df.loc[:, 'iatexplicitmath' + str(i)] = df.loc[:, 'iatexplicitmath' + str(i)].map(iatexplicitart_dict[i])
            return
        else:
            temp_df = df.copy(deep=True)
            for i in range(1, 7):
                temp_df.loc[:, 'iatexplicitart' + str(i)] = temp_df.loc[:, 'iatexplicitart' + str(i)].map(
                    iatexplicitart_dict[i])
                temp_df.loc[:, 'iatexplicitmath' + str(i)] = temp_df.loc[:, 'iatexplicitmath' + str(i)].map(
                    iatexplicitart_dict[i])
            return temp_df

    def clean_exprunafter2(self, df, inplace = False):
        """
        cleans exprunafter2 variable, and makes sensible groups ignoring different spellings.
        :param df: dataframe containing exprunafter2 variable
        :param inplace:
        :return: modified dataframe if inplace == False
        """
        exprunafter2_inverse_dict = {'intention': [x for x in df.exprunafter2.unique() if not pd.isnull(x) \
                                                   and 'intention' in x] + ['inentionality', 'intentionality'],
                                     'thinking and reasoning': [x for x in df.exprunafter2.unique() if not \
                                         pd.isnull(x) and 'thinking' in x],
                                     'emotion and verbal learning': [x for x in df.exprunafter2.unique() if not \
                                         pd.isnull(x) and 'emotion' in x],
                                     'understanding social communication': [x for x in df.exprunafter2.unique() \
                                                                            if not pd.isnull(x) and \
                                                                            'social' in x],
                                     'groups': [x for x in df.exprunafter2.unique() if not pd.isnull(x) and \
                                                'group' in x],
                                     'your past and future': ['your past and your future', \
                                                              '-your past and your future'],
                                     'linear regression lab': ['linear regression lab'],
                                     'it was not provided to me': ['it was not provided to me'],
                                     'no': ['no'],
                                     'verbal ospan': ['verbal ospan'],
                                     'trust game': ['trust game'],
                                     'a36': ['a36']}
        exprunafter2_dict = {val_i: key for key, val in exprunafter2_inverse_dict.items() for val_i in val}
        if inplace:
            df.exprunafter2 = df.exprunafter2.map(exprunafter2_dict)
            return
        else:
            temp_df = df.copy(deep=True)
            temp_df.exprunafter2 = temp_df.exprunafter2.map(exprunafter2_dict)
            return temp_df

    def make_flagsupplement_ordinal(self, df, inplace = False):
        """
        Makes flagsupplement1, flagsupplement2, flagsupplement3 variables ready for LabelEncoder
        ..deprecated - use sklearn.preprocessing.OrdinalEncoder instead
        :param df: dataframe containing flagsupplement1, flagsupplement2, flagsupplement3 variables
        :param inplace:
        :return: modified dataframe if inplace == False
        """
        flagsupplement1_dict = {'very much': '11_very_much',
                                'not at all': '1_not_at_all'}

        flagsupplement2_dict = {'republican': '7_republican',
                                'democrat': '1_democrat'}

        flagsupplement3_dict = {'conservative': '7_conservative',
                                'liberal': '1_liberal'}

        if inplace:
            df.loc[:, 'flagsupplement1'] = \
                df.loc[:, 'flagsupplement1'].apply(lambda x: x if (pd.isnull(x) or x not in \
                                                                   flagsupplement1_dict.keys()) else \
                    flagsupplement1_dict[x])
            df.loc[:, 'flagsupplement2'] = \
                df.loc[:, 'flagsupplement2'].apply(lambda x: x if (pd.isnull(x) or x not in \
                                                                   flagsupplement2_dict.keys()) else \
                    flagsupplement2_dict[x])
            df.loc[:, 'flagsupplement3'] = \
                df.loc[:, 'flagsupplement3'].apply(lambda x: x if (pd.isnull(x) or x not in \
                                                                   flagsupplement3_dict.keys()) else \
                    flagsupplement3_dict[x])
            return
        else:
            temp_df = df.copy(deep=True)
            temp_df.loc[:, 'flagsupplement1'] = \
                temp_df.loc[:, 'flagsupplement1'].apply(lambda x: x if (pd.isnull(x) or x not in \
                                                                        flagsupplement1_dict.keys()) else \
                    flagsupplement1_dict[x])
            temp_df.loc[:, 'flagsupplement2'] = \
                temp_df.loc[:, 'flagsupplement2'].apply(lambda x: x if (pd.isnull(x) or x not in \
                                                                        flagsupplement2_dict.keys()) else \
                    flagsupplement2_dict[x])
            temp_df.loc[:, 'flagsupplement3'] = \
                temp_df.loc[:, 'flagsupplement3'].apply(lambda x: x if (pd.isnull(x) or x not in \
                                                                        flagsupplement3_dict.keys()) else \
                    flagsupplement3_dict[x])
            return temp_df

    def make_sysjust_ordinal(self, df, inplace = False):
        """
        Makes sysjust[1-8] variables ready for LabelEncoder
        ..deprecated - use sklearn.preprocessing.OrdinalEncoder instead
        :param df: dataframe containing variables sysjust[1-8]
        :param inplace:
        :return: modified dataframe if inplace == False
        """
        sysjust_dict1 = {'strongly disagree': '1_strongly_disagree',
                         'strongly agree': '7_strongly_agree',
                         '2': '2', '3': '3', '4': '4', '5': '5', '6': '6'}
        sysjust_dict2 = {'strongly disagree': '7_strongly_disagree',
                         'strongly agree': '1_strongly_agree',
                         '2': '2', '3': '3', '4': '4', '5': '5', '6': '6'}
        if inplace:
            for col in ['sysjust' + str(i) for i in range(1, 9)]:
                if col in ['sysjust3', 'sysjust7']:
                    df.loc[:, col] = df.loc[:, col].map(sysjust_dict2)
                else:
                    df.loc[:, col] = df.loc[:, col].map(sysjust_dict1)
            return
        else:
            temp_df = df.copy(deep=True)
            for col in ['sysjust' + str(i) for i in range(1, 9)]:
                if col in ['sysjust3', 'sysjust7']:
                    temp_df.loc[:, col] = temp_df.loc[:, col].map(sysjust_dict2)
                else:
                    temp_df.loc[:, col] = temp_df.loc[:, col].map(sysjust_dict1)
            return temp_df

    def translate_text(self, df, group_fast_error=True, inplace = False):
        """
        translated text objects are stored on local disk. It retrieves those objects, which contain the
        translated object, the source language.
        21 rows have unidentified language, and they have been classified as np.NaN. Their indices are:
        [210, 211, 212, 213, 214, 217, 220, 221, 223, 231, 254, 255, 264,
                265, 270, 272, 276, 277, 282, 286, 287]
        TODO: Make sure that these rows are not classified as 'fast trials' or 'too many errors'.
        This function adds 'text_trans' and 'text_lang' variables to the dataframe
        :param df:
        :param group_fast_error: Should we group the 'fast trials' and 'too many errors' into one class
        of 'others'?
        :param inplace:
        :return: modified dataframe if inplace == False
        """
        import re
        def regex_moderate_art_math(x):
            return (len(re.findall('[a-z ]*moderate[a-z ]*art[a-z ]*math[a-z .]*', x)) > 0 or
                    len(re.findall('[a-z ]*(art more than mathematics)[a-z .]*', x)) > 0)

        def regex_other(x):
            return (len(
                re.findall('[a-z ?]*(st? öedn)[a-z ?]*(preference)[a-z ?]*(comparison)[a-z ?]*(mathematics.)', x)) > 0 or
                    len(re.findall('[a-z ?]*(st? öedn)[a-z ?]*(preference of mathematics)[a-z ?.]*', x)) > 0)

        def regex_neutral(x):
            return (len(re.findall('[a-z ]*(small|(little[a-z ]*no))[a-z .]*', x)) > 0 or \
                    len(re.findall('[a-z ]*no[a-z ]*(relevant|different)[a-z .]*', x)) > 0)

        def regex_fast(x):
            return len(re.findall('[a-z ]*(fast|quick)[a-z .]*', x)) > 0

        def regex_errors(x):
            return len(re.findall('[a-z ]*(mistake|error)[a-z .]*', x)) > 0

        def regex_strong_math_art(x):
            return (len(re.findall('[a-z ]*(strong)[a-z ]*math[a-z ]*art[a-z .]*', x)) > 0 or
                    len(re.findall('[a-z ]*(mathematics more strongly than art)[a-z .]*', x)) > 0)

        def regex_moderate_math_art(x):
            return (len(re.findall('[a-z ]*(moderate)[a-z ]*math[a-z ]*art[a-z .]*', x)) > 0 or
                    len(re.findall('[a-z ]*(mathematics more than art)[a-z .]*', x)) > 0)

        def regex_slight_math_art(x):
            return (len(re.findall('[a-z ]*(slight|weak)[a-z ]*math[a-z ]*art[a-z .]*', x)) > 0 or
                    len(re.findall('[a-z ]*(math a little bit more than art)[a-z .]*', x)) > 0 or
                    len(re.findall('[a-z? ]*(weak preference of mathematics)[a-z? .]*', x)) > 0 or
                    len(re.findall('[a-z ]*(prefer a little more than art math)[a-z .]*', x)) > 0)

        def regex_slight_art_math(x):
            return (len(re.findall('[a-z ]*(slight|weak)[a-z ]*art[a-z ]*math[a-z .]*', x)) > 0 or
                    len(re.findall('[a-z? ]*(weak preference)[a-z? ]*(in comparison)[a-z? ]*(math)[a-z? .]*', x)) > 0)

        def regex_strong_art_math(x):
            return (len(re.findall('[a-z ]*(strong)[a-z ]*art[a-z ]*math[a-z .]*', x)) > 0 or
                    len(re.findall('[a-x ]*(art more strongly than math)[a-z .]*', x)) > 0 or
                    len(re.findall('[a-x ]*(strong preference)[a-z? ]*(in comparison)[a-z? ]*(math)[a-z .]*', x)) > 0)

        import pickle
        with open(r'../ML1/text.pkl', 'rb') as f:
            s_temp = pickle.load(f)
        text_trans = pd.Series([t.text.lower() for t in s_temp])
        text_lang = pd.Series([t.src for t in s_temp])

        text_inverse_dict = {'1_your data suggests strong preference for arts compared to mathematics': \
                                 text_trans.loc[text_trans.map(regex_strong_art_math)].unique(),
                             '2_your data suggests moderate preference for arts compared to mathematics': \
                                 text_trans.loc[text_trans.map(regex_moderate_art_math)].unique(),
                             '3_your data suggests slight preference for arts compared to mathematics': \
                                 text_trans.loc[text_trans.map(regex_slight_art_math)].unique(),
                             '4_your data suggests little or no preference for mathematics compared arts': \
                                 text_trans.loc[text_trans.map(regex_neutral)].unique(),
                             '5_your data suggests slight preference for mathematics compared to arts': \
                                 text_trans.loc[text_trans.map(regex_slight_math_art)].unique(),
                             '6_your data suggests moderate preference for mathematics compared to arts': \
                                 text_trans.loc[text_trans.map(regex_moderate_math_art)].unique(),
                             '7_your data suggests strong preference for mathematics compared to arts': \
                                 text_trans.loc[text_trans.map(regex_strong_math_art)].unique(),
                             '.': ['.']}

        if group_fast_error:
            a = list(text_trans.loc[text_trans.map(regex_errors)].unique())
            b = list(text_trans.loc[text_trans.map(regex_fast)].unique())
            a.extend(b)
            text_inverse_dict['0_other'] = a
        else:
            text_inverse_dict['0_too many errors made to determine result'] = \
                text_trans.loc[text_trans.map(regex_errors)].unique()
            text_inverse_dict['0_too many fast trials conducted'] = \
                text_trans.loc[text_trans.map(regex_fast)].unique()

        text_dict = {val_i: key for key, val in text_inverse_dict.items() for val_i in val}
        text_dict['.'] = np.NaN

        if inplace:
            df.loc[:, 'text_trans'] = text_trans.map(text_dict)
            df.loc[:, 'text_lang'] = text_lang
            df.loc[df.query('text_trans != text_trans').index, 'text_lang'] = np.NaN
        else:
            temp_df = df.copy(deep=True)
            temp_df.loc[:, 'text_trans'] = text_trans.map(text_dict)
            temp_df.loc[:, 'text_lang'] = text_lang
            temp_df.loc[temp_df.query('text_trans != text_trans').index, 'text_lang'] = np.NaN
            return temp_df

    def preprocess_partgender(self, df, inplace = False):
        """
        Combines information from 'moneygendera', 'moneygenderb' and 'partgender' variables into
        'partgender'.
        :param df:
        :param inplace:
        :return:
        """
        if not self.__func_hist["put_nans"]:
            raise PreprocessorOrderError("Run put_nans function first")
        self.__func_hist["preprocess_partgender"] = True

        gender_dict = {'female': '1', 'male': '0'}
        gender_dict_reverse = {'1': 'female', '0': 'male'}
        if inplace:
            df.moneygendera = df.moneygendera.fillna(df.moneygenderb)
            df.moneygendera = df.moneygendera.fillna(df.partgender.map(gender_dict))
            df.partgender = df.partgender.fillna(df.moneygendera.map(gender_dict_reverse))
            df.moneygenderb = df.moneygenderb.fillna(df.moneygendera)
            return
        else:
            temp_df = df.copy(deep=True)
            temp_df.moneygendera = temp_df.moneygendera.fillna(temp_df.moneygenderb)
            temp_df.moneygendera = temp_df.moneygendera.fillna(temp_df.partgender.map(gender_dict))
            temp_df.partgender = temp_df.partgender.fillna(temp_df.moneygendera.map(gender_dict_reverse))
            temp_df.moneygenderb = temp_df.moneygenderb.fillna(temp_df.moneygendera)
            return temp_df

    def add_midnight(self, df, inplace = False):
        """
        Adds a midnight variable to our dataframe. We can detect such when creation_date is not equal
        to session date. Might/might not be useful.
        :param df:
        :param inplace:
        :return:
        """
        self.__func_hist["add_midnight"] = True

        if inplace:
            df.loc[:, 'midnight'] = "no"
            df.loc[df.query('creation_date != session_date'), 'midnight'] = "yes"
            return
        else:
            temp_df = df.copy(deep=True)
            temp_df.loc[:, 'midnight'] = "no"
            temp_df.loc[df.query('creation_date != session_date'):, 'midnight'] = "yes"
            return temp_df

    def convert_regression_col_str_float(self, df, column = None, inplace = False):
        """
        This function converts the passed column names' values to float.
        :param df:
        :param column: either a list of columns or a column_name string
        :param inplace:
        :return: modified dataframe if inplace == False
        """
        if not self.__func_hist["put_nans"]:
            raise PreprocessorOrderError("Run put_nans function first")

        if isinstance(column, list):
            self.cols['regression_cols'].extend(column)
            if inplace:
                df.loc[:, column] = df.loc[: column].applymap(float)
                return
            else:
                temp_df = df.copy(deep=True)
                temp_df.loc[:, column] = temp_df.loc[:, column].applymap(float)
                return temp_df
        else:
            self.cols['regression_cols'].append(column)
            if inplace:
                df.loc[:, column] = df.loc[: column].map(float)
                return
            else:
                temp_df = df.copy(deep=True)
                temp_df.loc[:, column] = temp_df.loc[:, column].map(float)
                return temp_df

    def remove_cols(self, df, column, type = 'remove', inplace = False):
        """
        Removes the specified column. Convenient for piping proprocessors
        :param df:
        :param column: List of strings of column indices, or string of column indices
        :param inplace:
        :return:
        """
        if type == 'remove':
            key = 'remove_cols'
        elif type == 'nlp':
            key = 'nlp_cols'

        if isinstance(column, list):
            self.cols[key].extend(column)
            if inplace:
                df.drop(labels = column, axis = 1, errors = 'ignore', inplace = False)
                return
            else:
                temp_df = df.drop(labels = column, axis = 1, errors = 'ignore', inplace = False)
                return temp_df
        else:
            self.cols[key].append(column)
            if inplace:
                df.drop(labels = column, axis = 1, inplace = True, errors = 'ignore')
                return
            else:
                temp_df = df.drop(labels = column, axis = 1, errors = 'ignore', inplace = False)
                return temp_df

    def remove_single_value_cols(self, df, inplace = False):
        """
        Removes all the columns which have no missing entries, and only a single unique value.
        :param df:
        :param inplace:
        :return:
        """
        if not self.__func_hist["put_nans"]:
            raise PreprocessorOrderError("Run put_nans function first")

        self.__func_hist['remove_single_value_cols'] = True

        self.cols['single_value_cols'].extend([col for col in df.columns \
                                               if df.loc[:, col].unique().shape[0] == 1])
        if inplace:
            self.remove_cols(df, self.cols['single_value_cols'], inplace = True)
            return
        else:
            return self.remove_cols(df, self.cols['single_value_cols'], inplace = False)

    def add_omdim_mean_col(self, df, inplace = False):
        """
        Since omdimc3rt and omdimc3trt follow each other very closely, we can combine the information
        of both the variables by adding the mean of both as 'omdimc3rt_omdimc3trt_mean'
        :param df:
        :param inplace:
        :return:
        """
        if any([df.loc[:, col].dtype == 'O' for col in ['omdimc3rt', 'omdimc3trt']]):
            raise PreprocessorOrderError("omdimc3rt and omdimc3trt are not converted to float")

        if inplace:
            df.loc[:, 'omdimc3rt_omdimc3trt_mean'] = df.loc[:, ['omdimc3rt', 'omdimc3trt']].\
                mean(axis = 1)
            return
        else:
            temp_df = df.copy(deep = True)
            temp_df.loc[:, 'omdimc3rt_omdimc3trt_mean'] = \
                temp_df.loc[:, ['omdimc3rt', 'omdimc3trt']].mean(axis = 1)
            return temp_df

    def create_power_set(self, df, max_row_null_per = 10, max_col_null_per = 20,
                         inplace = False):
        """
        A loop to create a power set of rows and columns from our data,
        which shall have rows with at max max_row_null_per % of entries as null
        and shall have columns with at max max_col_null_per % of entries as null.

        :param df:
        :param max_row_null_per:
        :param max_col_null_per:
        :param inplace:
        :return:
        """
        if not self.__func_hist["put_nans"]:
            raise PreprocessorOrderError("Run put_nans function first")
        if not self.__func_hist["remove_single_value_cols"]:
            raise PreprocessorOrderError("Run remove_single_value_cols function first")
        self.__func_hist["create_power_set"] = True

        flag = 0

        if inplace:
            while flag == 0:
                flag = 1
                no_rows = df.shape[0]
                remove_col = []
                for col_name, col in df.iteritems():
                    if list(pd.isnull(col)).count(True) * 100 // no_rows > max_col_null_per:
                        remove_col.append(col_name)
                        flag = 0
                df.drop(remove_col, inplace=True, axis=1)
                self.cols['power_set_removed_cols'].extend(remove_col)

                no_cols = df.shape[1]
                remove_rows = []
                for index, row in df.iterrows():
                    if list(pd.isnull(row)).count(True) * 100 // no_cols > max_row_null_per:
                        remove_rows.append(index)
                        flag = 0
                df.drop(remove_rows, axis=0, inplace=True)
                self.cols['power_set_removed_rows'].extend(remove_rows)
            return
        else:
            temp_df = df.copy(deep=True)
            while flag == 0:
                flag = 1
                no_rows = temp_df.shape[0]
                remove_col = []
                for col_name, col in temp_df.iteritems():
                    if list(pd.isnull(col)).count(True) * 100 // no_rows > max_col_null_per:
                        remove_col.append(col_name)
                        flag = 0
                temp_df.drop(remove_col, inplace=True, axis=1)
                self.cols['power_set_removed_cols'].extend(remove_col)

                no_cols = temp_df.shape[1]
                remove_rows = []
                for index, row in temp_df.iterrows():
                    if list(pd.isnull(row)).count(True) * 100 // no_cols > max_row_null_per:
                        remove_rows.append(index)
                        flag = 0
                temp_df.drop(remove_rows, axis=0, inplace=True)
                self.cols['power_set_removed_rows'].extend(remove_rows)
            return temp_df

    def fit_transform_label_encoder(self, df, columns, inplace = False):
        """
        Creates a label encoder for each column in columns, and stores them in a map
        for inverse transform. This function just removes the responsibility from users of
        keeping track of labelEncoders for each column.
        :param df:
        :param columns: list or String
        :param inplace:
        :return:
        """
        if not self.__func_hist["put_nans"]:
            raise PreprocessorOrderError("Run put_nans function first")
        self.__func_hist['fit_transform_label_encoder'] = True

        from sklearn import preprocessing
        if isinstance(columns, list):
            if inplace:
                for col in columns:
                    le = preprocessing.LabelEncoder()
                    le.fit(df.loc[:, col])
                    df.loc[:, col] = le.transform(df.loc[:, col])
                    self.label_encoders[col] = le
            else:
                temp_df = df.copy(deep=True)
                for col in columns:
                    le = preprocessing.LabelEncoder()
                    le.fit(df.loc[:, col])
                    temp_df.loc[:, col] = le.transform(df.loc[:, col])
                    self.label_encoders[col] = le
                return temp_df
        else:
            le = preprocessing.LabelEncoder()
            le.fit(df.loc[:, columns])
            if inplace:
                df.loc[:, columns] = le.transform(df.loc[:, columns])
                self.label_encoders[columns] = le
            else:
                temp_df = df.copy(deep=True)
                temp_df.loc[:, columns] = le.transform(df.loc[:, columns])
                return temp_df
