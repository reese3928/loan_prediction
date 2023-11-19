import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import json
import numpy as np

class LoanPrediction(object):
    '''
    A class used to load loan prediction data, build prediction model and evaluate output
    '''

    def __init__(self):
        self.raw_data = None
    
    def load_config(self, config):
        '''
        Load configuration file
        '''
        f = open(config)
        json_input = json.load(f)
        self.json_input = json_input
        self.data_path = json_input["data_path"]
        self.input_file_name = json_input["input_file_name"]
        self.res_path = json_input["res_path"]
        self.column_filter_missing = json_input["column_filter_missing"]
        self.employment_length_col = json_input["employment_length_col"]
        self.home_ownership_col = json_input["home_ownership_col"]
        self.home_ownership_categories = json_input["home_ownership_categories"]
        self.bad_loan_status = json_input["bad_loan_status"]
        self.loan_status_col = json_input["loan_status_col"]
        self.continuous_features = json_input["continuous_features"]
        self.discrete_features = json_input["discrete_features"]
        self.credit_history_column = json_input["credit_history_column"]

    def read_data(self):
        '''
        Read input file
        '''
        input_file = os.path.join(self.data_path, self.input_file_name)
        self.raw_data = pd.read_csv(input_file)

    def preprocess_data(self):
        df = self.raw_data

        # filter out missing values
        df_sub = df[self.column_filter_missing].dropna()
        print('Raw data shape: ', df.shape)
        print('After drop missing: ', df_sub.shape)

        # convert employment length into continuous feature
        df_sub[self.employment_length_col] = df_sub[self.employment_length_col].map({
            '10+ years': 10,
            '9 years': 9,
            '8 years': 8,
            '7 years': 7,
            '6 years': 6,
            '5 years': 5,
            '4 years': 4,
            '3 years': 3,
            '2 years': 2,
            '1 year': 1,
            '< 1 year': 0
        })

        # drop unnecessary home_ownership categories such that it can be one hot encoded later
        df_sub = df_sub[df_sub[self.home_ownership_col].isin(self.home_ownership_categories)]

        # calculate credit history in months
        df_sub[self.credit_history_column] = pd.to_datetime(df_sub[self.credit_history_column])
        df_sub[self.credit_history_column + '_mths'] = (pd.to_datetime('2023-11-18') - df_sub[self.credit_history_column]) / np.timedelta64(1, 'M')
        df_sub = df_sub[df_sub[self.credit_history_column + '_mths'] >= 0]

        self.processed_data = df_sub
    
    def data_exploration(self):
        '''
        For continuous features plot their histogram to get an idea of their distributions.
        For discrete features, count their values in each category.
        '''

        # if the directory does not exist yet, create one
        if not os.path.exists(self.res_path):
            os.mkdir(self.res_path)
        
        res_figure_path = os.path.join(self.res_path, 'figures')
        res_table_path = os.path.join(self.res_path, 'tables')

        if not os.path.exists(res_figure_path):
            os.mkdir(res_figure_path)

        if not os.path.exists(res_table_path):
            os.mkdir(res_table_path)

        # Save the distribution of continuous features
        for col in self.continuous_features:
            plt.figure(figsize=(10,6))
            plt.hist(self.processed_data[col])
            plt.xlabel(col)
            plt.title(f'Histogram of {col}')
            plt.savefig(os.path.join(res_figure_path, f'hist_{col}.png'), dpi=300)
            plt.close()

        # Save the distribution of discrete features
        for col in self.discrete_features:
            temp_value_count = self.processed_data[col].value_counts().to_frame()
            temp_value_count.to_csv(os.path.join(res_table_path, f'Distribution of {col}.csv'))

    def train_and_test_model(self, clf, grid_search_params):
        self.clf = clf

        # convert loan_status into a binary variable. This will be the prediction target
        self.processed_data['y'] = 0
        self.processed_data.loc[self.processed_data[self.loan_status_col].isin(self.bad_loan_status), 'y'] = 1

        y = self.processed_data['y']
        X = self.processed_data[self.continuous_features + self.discrete_features]
        # one hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # split data into 70% training and 30% testing
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        self.x_train = x_train 
        self.x_test = x_test 
        self.y_train = y_train 
        self.y_test = y_test

        # train model and run cross validation to select the optimal hyperparameters
        # Grid Search Parameters
        clf_best = GridSearchCV(estimator=clf, param_grid=grid_search_params, cv=5)
        clf_best.fit(self.x_train, self.y_train)
        self.clf_best = clf_best.best_estimator_

        # make prediction on the test data
        self.predictions = clf_best.predict(self.x_test)
        self.prediction_score = clf_best.predict_proba(self.x_test)
    
    def model_eval(self, model_name):

        # calculate metrics
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions)
        recall = recall_score(self.y_test, self.predictions)
        f1score = f1_score(self.y_test, self.predictions)
        auc_score = roc_auc_score(self.y_test, self.prediction_score[:,1])

        # save metrics into a table
        temp_df = pd.DataFrame({
            "metrics": ['accuracy', 'precision', 'recall', 'f1score', 'auc'],
            "value": [accuracy, precision, recall, f1score, auc_score]
        })
        res_table_path = os.path.join(self.res_path, 'tables')
        temp_df.to_csv(os.path.join(res_table_path, f'eval_metrics_{model_name}.csv'), index=False)

        # save visulization into table
        temp_df = pd.DataFrame({
            "prediction_score": self.prediction_score[:,1],
            "label": self.y_test
        })

        plt.figure(figsize=(10,6))
        res_figure_path = os.path.join(self.res_path, 'figures')
        temp_df.boxplot(column='prediction_score', by='label')
        plt.title(f'{model_name}')
        plt.ylabel('prediction_score')
        plt.savefig(os.path.join(res_figure_path, f'prediction_score_{model_name}.png'), dpi=300)
        plt.close()



