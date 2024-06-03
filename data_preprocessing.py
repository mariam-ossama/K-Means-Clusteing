import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import zscore

class DataProcessor:
    def __init__(self, file_path, columns_to_drop=None):
        self.file_path = file_path
        self.columns_to_drop = columns_to_drop
        self.data = pd.read_csv(file_path)
        print(len(self.data))
        if self.columns_to_drop:
            self.data = self.data.drop(columns=self.columns_to_drop)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def preprocess_data(self, percentage=100, remove_outliers=True):
        # Sample a percentage of the data
        if percentage < 100:
            sampled_indices = np.random.choice(len(self.data), size=int(len(self.data) * percentage / 100),
                                               replace=False)
            self.data = self.data.iloc[sampled_indices]
            print(len(self.data))

            # Remove duplicates
            self.data = self.data.drop_duplicates()

        # Handle missing values in numeric columns
        numeric_data = self.data.select_dtypes(include=np.number)
        imputer_numeric = SimpleImputer(strategy='mean')
        numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(numeric_data), columns=numeric_data.columns)

        # Handle missing values in non-numeric columns
        non_numeric_data = self.data.select_dtypes(exclude=np.number)
        imputer_non_numeric = SimpleImputer(strategy='constant', fill_value='Missing')
        non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(non_numeric_data),
                                           columns=non_numeric_data.columns)

        # Combine numeric and non-numeric data
        df_imputed = pd.concat([non_numeric_imputed, numeric_imputed], axis=1)

        # Ensure all columns are numeric
        df_imputed = df_imputed.apply(pd.to_numeric, errors='ignore')

        # Identify categorical columns except 'Movie Name'
        categorical_cols = [col for col in df_imputed.columns if
                            col != 'Movie Name' and df_imputed[col].dtype == 'object']

        # Encode categorical variables
        if categorical_cols:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            df_encoded = pd.DataFrame(encoder.fit_transform(df_imputed[categorical_cols]))
            df_encoded.columns = encoder.get_feature_names_out(categorical_cols)
            df_final = pd.concat([df_imputed.drop(columns=categorical_cols), df_encoded], axis=1)
        else:
            df_final = df_imputed.copy()

        # Calculate z-scores only for numeric columns
        numeric_columns = df_final.select_dtypes(include=np.number).columns
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df_final[numeric_columns].quantile(0.25)
        Q3 = df_final[numeric_columns].quantile(0.75)

        # Calculate IQR (Interquartile Range)
        IQR = Q3 - Q1

        # Define lower and upper bounds for outlier detection
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"lower bound:\n {lower_bound}")
        print(f"upper bound:\n {upper_bound}")

        # Identify outliers
        outliers = ((df_final[numeric_columns] < lower_bound) | (df_final[numeric_columns] > upper_bound)).any(axis=1)

        # Remove outliers
        df_no_outliers = df_final[~outliers]

        # Reset index of the DataFrame
        df_no_outliers.reset_index(drop=True, inplace=True)

        outliers_data = df_imputed.reset_index(drop=True)[outliers][['Movie Name', 'Duration']]  # Extract outlier movie names and ratings
        outliers_list = outliers_data.values.tolist()  # Convert outliers data to a list of lists

        return df_no_outliers, outliers_list


    def _handle_missing_and_encode(self):
        # Handle missing values in numeric columns
        numeric_data = self.data.select_dtypes(include=np.number)
        imputer_numeric = SimpleImputer(strategy='mean')
        numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(numeric_data), columns=numeric_data.columns)

        # Handle missing values in non-numeric columns
        non_numeric_data = self.data.select_dtypes(exclude=np.number)
        imputer_non_numeric = SimpleImputer(strategy='mode')
        non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(non_numeric_data),
                                           columns=non_numeric_data.columns)

        # Combine numeric and non-numeric data
        df_imputed = pd.concat([non_numeric_imputed, numeric_imputed], axis=1)

        # Encode categorical variables
        categorical_cols = [col for col in df_imputed.columns if col != 'Movie Name' and df_imputed[col].dtype == 'object']
        if categorical_cols:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            df_encoded = pd.DataFrame(encoder.fit_transform(df_imputed[categorical_cols]))
            df_encoded.columns = encoder.get_feature_names_out(categorical_cols)
            df_final = pd.concat([df_imputed.drop(columns=categorical_cols), df_encoded], axis=1)
        else:
            df_final = df_imputed.copy()

        return df_final

    def train_test_split(self, target_column, test_size=0.2, random_state=None):
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(len(self.data))
        test_set_size = int(len(self.data) * test_size)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        self.X_train = self.data.iloc[train_indices].drop(columns=target_column)
        self.X_test = self.data.iloc[test_indices].drop(columns=target_column)
        self.y_train = self.data.iloc[train_indices][target_column]
        self.y_test = self.data.iloc[test_indices][target_column]

    '''def _detect_outliers(self, df):
        z_scores = np.abs(zscore(df))
        threshold = 3
        return ((z_scores >= threshold).any(axis=1))'''
