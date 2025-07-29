import pandas as pd
from sklearn.model_selection import train_test_split
from core.utils import cleanup_data, normalize_data
import pprint

class BaseMLModels:
    def __init__(self):
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_dataset(self, label, drop_col, filename, train_size=0.8, val_size=0.1, test_size=0.1):
        """
        Load data from a CSV file.
        :param filename: Path to the CSV file.
        :return: DataFrame containing the loaded data.
        """
        df = pd.read_csv(filename)
        df = self.cleanup_data(df)
        df = self.normalize_data(df)
        target = df[label]
        if target.dtype == 'category':
            target, _ = pd.factorize(target)
        df = df.drop(columns=drop_col, errors='ignore')

        # Split data into training, validation, and test sets
        X, X_test, y, y_test = train_test_split(df,
                                                target,
                                                test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          test_size=val_size*train_size, random_state=42)
        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)

    def train_model(self):
        """
        Train the XGBoost model with the training data.
        :param params: Dictionary of parameters for the XGBoost model.
        """
        self.model.fit(self.train_data[0], self.train_data[1],
                       eval_set=[self.val_data], verbose=True)
        print("Model trained successfully.")
        pprint.pprint(self.model.evals_result())

    def save_model(self, filename):
        """
        Save the trained model to a file.
        :param filename: Path to save the model.
        """
        self.model.save_model(filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """
        Load a model from a file.
        :param filename: Path to the model file.
        """
        self.model.load_model(filename)

    def predict(self, X):
        """
        Make predictions using the trained model.
        :param X: DataFrame or array-like structure containing the features for prediction.
        :return: Predictions made by the model.
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        return self.model.predict(X)

    def get_columns(self):
        """
        Get the feature names used in the model.
        :return: List of feature names.
        """
        if self.train_data is not None:
            return self.train_data[0].columns.tolist()
        else:
            raise ValueError("Training data is not loaded.")
