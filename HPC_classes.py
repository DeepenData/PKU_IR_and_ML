from pyspark.sql import DataFrame
from pyspark.sql.functions import col, regexp_replace, when
from pyspark.sql.types import FloatType

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, regexp_replace

def drop_columns_containing_word(df, word, exception=None):
    """
    Drop columns from a PySpark DataFrame if their names contain a specific word, 
    unless the column name matches the provided exception.

    Parameters:
    df (pyspark.sql.DataFrame): The DataFrame from which to drop columns.
    word (str): The word to look for in column names.
    exception (str, optional): The column name that should not be dropped even if it contains the word.

    Returns:
    df (pyspark.sql.DataFrame): The DataFrame after dropping the columns.
    """
    for col in df.columns:
        if word.lower() in col.lower() and col != exception:
            df = df.drop(col)

    return df



class pysparkDataFrameTransformer:
    """
    A utility class that provides methods to perform transformations on a PySpark DataFrame.
    This class supports chaining of methods for a fluent-like API.

    Attributes:
    data_frame (DataFrame): The PySpark DataFrame to be transformed.
    """

    def __init__(self, data_frame: DataFrame):
        """
        Initializes a new instance of the DataFrameTransformer class.

        Parameters:
        data_frame (DataFrame): The PySpark DataFrame to be transformed.
        """
        self.data_frame = data_frame

    def replace_with_regex(self, column_names: list, pattern: str, replacement: str):
        """
        Replaces occurrences of a regular expression pattern in specified columns with a replacement string.

        Parameters:
        column_names (list): The names of the columns in which to replace the pattern.
        pattern (str): The regular expression pattern to replace.
        replacement (str): The string to use as the replacement.

        Returns:
        pysparkDataFrameTransformer: The instance of the DataFrameTransformer with updated DataFrame.
        """
        for column_name in column_names:
            self.data_frame = self.data_frame.withColumn(column_name, 
                                                         regexp_replace(col(column_name), pattern, replacement))
        return self

    def replace_non_numeric_with_float(self, column_names: list, replacement: float):
        """
        Replaces non-numeric entries in specified columns with a replacement float.

        Parameters:
        column_names (list): The names of the columns in which to replace non-numeric entries.
        replacement (float): The float to use as the replacement.

        Returns:
        pysparkDataFrameTransformer: The instance of the DataFrameTransformer with updated DataFrame.
        """
        for column_name in column_names:
            self.data_frame = self.data_frame.withColumn(
                column_name,
                when(col(column_name).cast("float").isNotNull(), col(column_name).cast("float")).otherwise(replacement)
            )
        return self

    def remove_rows_with_missing_values(self, column_names: list):
        """
        Removes rows from the DataFrame that contain missing values (NA, NAN) in any of the specified columns.

        Parameters:
        column_names (list): The names of the columns in which to look for missing values.

        Returns:
        pysparkDataFrameTransformer: The instance of the DataFrameTransformer with updated DataFrame.
        """
        for column_name in column_names:
            self.data_frame = self.data_frame.filter(col(column_name).isNotNull())
        return self

    def binarize_column(self, column_names: list, cut_off: float, new_column_names: list = None, drop_original: bool = False):
        """
        Creates new categorical columns based on a cut-off value.
        The new columns will have entries 'below' if the corresponding entry in the input column is below the cut-off, 
        and 'above' if the entry is equal to or greater than the cut-off.

        Parameters:
        column_names (list): The names of the input columns.
        cut_off (float): The cut-off value.
        new_column_names (list, optional): The names of the new binarized columns. Defaults to None.
        drop_original (bool, optional): Whether to drop the original columns. Defaults to False.

        Returns:
        pysparkDataFrameTransformer: The instance of the DataFrameTransformer with updated DataFrame.
        """
        for i, column_name in enumerate(column_names):
            new_column_name = new_column_names[i] if new_column_names else f"binarized_{column_name}"
            self.data_frame = self.data_frame.withColumn(
                new_column_name, 
                when(col(column_name) < cut_off, str(0)).otherwise(str(1))
            )
            if drop_original:
                self.data_frame = self.data_frame.drop(column_name)
        return self


import json
import random
import urllib

import numpy as np
import optuna
import pandas as pd
import ray
import requests
import shap
import xgboost
import yaml

from pyspark.sql import SparkSession
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

CPUS_PER_JOB : int = 5

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



class DataImputer:
    """
    A class to impute missing numerical values in a pandas DataFrame.

    Attributes
    ----------
    df : pandas.DataFrame
        The input dataframe with data.
    random_state : int
        The seed used by the random number generator for the imputer.
    max_iter : int
        The maximum number of imputing iterations to perform.
    scaler : sklearn.StandardScaler
        The standard scaler object.
    imputer : sklearn.IterativeImputer
        The iterative imputer object.

    Methods
    -------
    fit_transform():
        Fits the imputer on data and returns the imputed DataFrame.
    insert_random_nans(probability: float = 0.2):
        Inserts random NaNs to numerical columns in the dataframe.
    """

    def __init__(self, df, random_state=None, max_iter=10):
        """
        Constructs all the necessary attributes for the DataImputer object.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe with data.
        random_state : int, optional
            The seed used by the random number generator for the imputer (default is 100).
        max_iter : int, optional
            The maximum number of imputing iterations to perform (default is 10).
        """
        self.df = df
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.imputer = IterativeImputer(random_state=self.random_state, max_iter=self.max_iter)

    def fit_transform(self):
        """
        Fits the imputer on data and performs imputations, and then inverse transform to retain the original scale.

        Returns
        -------
        DataImputer
            The instance of the DataImputer.
        """
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        numerical_df = self.df[numerical_cols]

        # Scale numerical data
        scaled_df = pd.DataFrame(self.scaler.fit_transform(numerical_df.values), columns=numerical_df.columns)

        # Fit imputer and perform imputations on the scaled numerical dataset
        df_imputed = pd.DataFrame(self.imputer.fit_transform(scaled_df), columns=scaled_df.columns)

        # Inverse transform to retain the original scale
        original_scale_df = pd.DataFrame(self.scaler.inverse_transform(df_imputed), columns=df_imputed.columns)

        # Combine imputed numerical data with the categorical data
        categorical_cols = self.df.select_dtypes(exclude=np.number).columns
        df_imputed_with_categorical = pd.concat([original_scale_df, self.df[categorical_cols]], axis=1)

        self.df = df_imputed_with_categorical

        return self

    def insert_random_nans(self, probability : float = 0.2):
        """
        Inserts random NaNs to numerical columns in the dataframe with a specified probability.

        Parameters
        ----------
        probability : float, optional
            The probability of a value being replaced with NaN (default is 0.2).

        Returns
        -------
        DataImputer
            The instance of the DataImputer.
        """
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        df_with_nans = self.df.copy()

        np.random.seed(self.random_state)

        for col in numerical_cols:
            mask = np.random.choice([False, True], size=len(df_with_nans), p=[1 - probability, probability])
            df_with_nans.loc[mask, col] = np.nan

        self.df = df_with_nans

        return self


def encode_and_convert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical variables and converts all data to float type.

    Parameters:
        df (pd.DataFrame): The DataFrame to be processed.

    Returns:
        df (pd.DataFrame): The processed DataFrame with categorical variables encoded and all data converted to float type.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype('category').cat.codes.astype('category')
        else:
            df[col] = df[col].astype('float')

    return df



class SparkDataProcessor:
    """
    This class is responsible for processing data using Apache Spark. It provides methods for data cleaning, 
    encoding categorical features and converting data types.

    Attributes:
        spark (SparkSession): The SparkSession object.
    """

    def __init__(self):
        """Initialize SparkDataProcessor with a SparkSession object."""
        self.spark = SparkSession.builder.appName("pku").master("local[*]").getOrCreate()

    @staticmethod
    def encode_and_convert(df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical variables and converts all data to float type.

        Parameters:
            df (pd.DataFrame): The DataFrame to be processed.

        Returns:
            df (pd.DataFrame): The processed DataFrame with categorical variables encoded and all data converted to float type.
        """
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype('category').cat.codes.astype('category')
            else:
                df[col] = df[col].astype('float')

        return df

    def load_file(self, url=None, path=None):
        """
        Retrieves a file content from either a URL or a local file path.

        Parameters:
            url (str, optional): URL from where to retrieve the file content.
            path (str, optional): Local file path from where to retrieve the file content.

        Returns:
            content (str): The file content.
        """
        if url:
            response = requests.get(url)
            content = response.text
        elif path:
            with open(path, 'r') as file:
                content = file.read()
        else:
            raise ValueError('Either a URL or a local file path needs to be provided.')

        return content

    def process_data(self, data_csv_url=None, data_csv_path=None, drop_col_yml_url=None, drop_col_yml_path=None, rename_col_json_url=None, rename_col_json_path=None):
        """
        Process the data by removing, renaming, and transforming columns. The method handles missing values and 
        converts the PySpark DataFrame to a pandas DataFrame.

        Parameters:
            data_csv_url (str, optional): The URL from where to retrieve the CSV data file.
            data_csv_path (str, optional): The local file path from where to retrieve the CSV data file.
            drop_col_yml_url (str, optional): The URL from where to retrieve the parameters file in YAML format.
            drop_col_yml_path (str, optional): The local file path from where to retrieve the parameters file in YAML format.
            rename_col_json_url (str, optional): The URL from where to retrieve the rename mapping file in JSON format.
            rename_col_json_path (str, optional): The local file path from where to retrieve the rename mapping file in JSON format.

        Returns:
            df (pd.DataFrame): The processed DataFrame.

        Raises:
            ValueError: If neither URL nor local file path is provided for any input file.
            ValueError: If loading content from any input file fails.
        """

        # Ensure that either URL or local path is provided for each input file
        if not data_csv_url and not data_csv_path:
            raise ValueError('Either data_csv_url or data_csv_path must be provided.')
        
        # Load CSV data into Spark DataFrame
        csv_file_path = data_csv_path or urllib.request.urlretrieve(data_csv_url, filename="/tmp/data.csv")[0]
        df = self.spark.read.csv(csv_file_path, inferSchema=True, header=True)

        # Load parameters and rename mapping files
        drop_col_yml_content = self.load_file(drop_col_yml_url, drop_col_yml_path) if drop_col_yml_url or drop_col_yml_path else None
        rename_col_json_content = self.load_file(rename_col_json_url, rename_col_json_path) if rename_col_json_url or rename_col_json_path else None

        # Parse parameters and rename mapping
        params = yaml.safe_load(drop_col_yml_content) if drop_col_yml_content else None
        rename_dict = json.loads(rename_col_json_content) if rename_col_json_content else None

        # Drop columns as specified in the parameters file
        if params and "feature_engineering" in params and "removed_features" in params["feature_engineering"]:
            df = df.drop(*params["feature_engineering"]["removed_features"])

        # Rename columns based on the rename mapping
        if rename_dict:
            for old_name, new_name in rename_dict.items():
                df = df.withColumnRenamed(old_name, new_name)

        return self.encode_and_convert(df.toPandas()), df