import pandas as pd  # Importing pandas for data handling
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting data
from sklearn.linear_model import LogisticRegression, LinearRegression  # Importing Logistic Regression and Linear Regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # Importing Decision Tree Classifier and Regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor  # Importing Random Forest and AdaBoost
from sklearn.preprocessing import LabelEncoder  # Importing LabelEncoder for encoding categorical target variables
from sklearn.impute import SimpleImputer  # Importing SimpleImputer for handling missing values

class MLEngine:
    def __init__(self, algorithm, csv_path, target_column):
        """
        Initialize the MLEngine with the selected algorithm, CSV path, and target column.
        """
        try:
            self.algorithm = algorithm
            self.csv_path = csv_path
            self.target_column = target_column
            self.data = pd.read_csv(self.csv_path)
            self.data = self._preprocess_data(self.data)
            self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
            self.model = self._select_model()
        except Exception as e:
            print(f"Error initializing MLEngine: {e}")

    def _preprocess_data(self, data):
        """
        Preprocess the data by handling missing values and encoding categorical features.
        """
        try:
            # Handling missing values by imputing them with the mean of the column for numeric columns
            imputer = SimpleImputer(strategy='mean')
            for column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    if data[column].isnull().any():
                        data[column] = imputer.fit_transform(data[column].values.reshape(-1, 1)).ravel()

            # Encoding categorical features
            encoded_data = pd.get_dummies(data)

            return encoded_data
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return data

    def _prepare_data(self):
        """
        Prepare the data by selecting the target column and splitting it into training and testing sets.
        """
        try:
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]

            # Determine if the target is categorical or continuous
            if y.dtype == 'object' or len(set(y)) <= 10:
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                return train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                return train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None, None, None

    def _select_model(self):
        """
        Select the machine learning model based on the specified algorithm.
        """
        try:
            is_classification = self.y_train is not None and len(set(self.y_train)) <= 10
            
            if self.algorithm == "Logistic Regression":
                return LogisticRegression() if is_classification else LinearRegression()
            elif self.algorithm == "Decision Tree":
                return DecisionTreeClassifier() if is_classification else DecisionTreeRegressor()
            elif self.algorithm == "Random Forest":
                return RandomForestClassifier() if is_classification else RandomForestRegressor()
            elif self.algorithm == "AdaBoost":
                return AdaBoostClassifier() if is_classification else AdaBoostRegressor()
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
        except Exception as e:
            print(f"Error selecting model: {e}")
            return None

    def run(self):
        """
        Run the selected model and return the results including accuracy and predictions.
        """
        try:
            if self.model is None:
                raise ValueError("No model has been selected.")

            self.model.fit(self.X_train, self.y_train)
            
            if hasattr(self.model, 'score'):
                accuracy = self.model.score(self.X_test, self.y_test)
            else:
                accuracy = None

            predictions = self.model.predict(self.X_test)
            return {"Algorithm": self.algorithm, "Accuracy": accuracy, "Predictions": predictions}
        except Exception as e:
            print(f"Error running model: {e}")
            return {"Algorithm": self.algorithm, "Accuracy": 0, "Predictions": []}
