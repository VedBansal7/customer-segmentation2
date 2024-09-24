import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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

            # Ensure target column exists, if not, raise an error
            if self.target_column not in self.data.columns:
                raise ValueError(f"'{self.target_column}' not found in the dataset")

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
                raise ValueError("Invalid algorithm specified.")
        except Exception as e:
            print(f"Error selecting model: {e}")
            return None

    def train_model(self):
        """
        Train the selected machine learning model.
        """
        try:
            if self.model is not None:
                self.model.fit(self.X_train, self.y_train)
                print("Model training completed.")
            else:
                print("Model is not initialized.")
        except Exception as e:
            print(f"Error training the model: {e}")

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        """
        try:
            if self.model is not None:
                score = self.model.score(self.X_test, self.y_test)
                print(f"Model evaluation score: {score}")
            else:
                print("Model is not initialized.")
        except Exception as e:
            print(f"Error evaluating the model: {e}")

    def run(self):
        """
        Train the model, evaluate it, and return the results.
        """
        try:
            self.train_model()
            self.evaluate_model()
            
            # Make predictions on the test set
            predictions = self.model.predict(self.X_test)
            
            return {'Predictions': predictions}  # Return predictions
        except Exception as e:
            print(f"Error running model: {e}")
            return None

    def predict(self, new_data):
        """
        Predict using the trained model on new data.
        """
        try:
            if self.model is not None:
                new_data_preprocessed = self._preprocess_data(new_data)
                predictions = self.model.predict(new_data_preprocessed)
                return predictions
            else:
                print("Model is not initialized.")
                return None
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None




























# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import SimpleImputer

# class MLEngine:
#     def __init__(self, algorithm, csv_path, target_column):
#         """
#         Initialize the MLEngine with the selected algorithm, CSV path, and target column.
#         """
#         try:
#             self.algorithm = algorithm
#             self.csv_path = csv_path
#             self.target_column = target_column
#             self.data = pd.read_csv(self.csv_path)

#             # Ensure target column exists, if not, raise an error
#             if self.target_column not in self.data.columns:
#                 raise ValueError(f"'{self.target_column}' not found in the dataset")

#             self.data = self._preprocess_data(self.data)
#             self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
#             self.model = self._select_model()
#         except Exception as e:
#             print(f"Error initializing MLEngine: {e}")

#     def _preprocess_data(self, data):
#         """
#         Preprocess the data by handling missing values and encoding categorical features.
#         """
#         try:
#             # Handling missing values by imputing them with the mean of the column for numeric columns
#             imputer = SimpleImputer(strategy='mean')
#             for column in data.columns:
#                 if data[column].dtype in ['int64', 'float64']:
#                     if data[column].isnull().any():
#                         data[column] = imputer.fit_transform(data[column].values.reshape(-1, 1)).ravel()

#             # Encoding categorical features
#             encoded_data = pd.get_dummies(data)

#             return encoded_data
#         except Exception as e:
#             print(f"Error preprocessing data: {e}")
#             return data

#     def _prepare_data(self):
#         """
#         Prepare the data by selecting the target column and splitting it into training and testing sets.
#         """
#         try:
#             X = self.data.drop(columns=[self.target_column])
#             y = self.data[self.target_column]

#             # Determine if the target is categorical or continuous
#             if y.dtype == 'object' or len(set(y)) <= 10:
#                 le = LabelEncoder()
#                 y = le.fit_transform(y.astype(str))
#                 self.product_names = list(le.classes_)  # Save the mapping of numerical to product names
#                 return train_test_split(X, y, test_size=0.2, random_state=42)
#             else:
#                 return train_test_split(X, y, test_size=0.2, random_state=42)
#         except Exception as e:
#             print(f"Error preparing data: {e}")
#             return None, None, None, None

#     def _select_model(self):
#         """
#         Select the machine learning model based on the specified algorithm.
#         """
#         try:
#             is_classification = self.y_train is not None and len(set(self.y_train)) <= 10
            
#             if self.algorithm == "Logistic Regression":
#                 return LogisticRegression() if is_classification else LinearRegression()
#             elif self.algorithm == "Decision Tree":
#                 return DecisionTreeClassifier() if is_classification else DecisionTreeRegressor()
#             elif self.algorithm == "Random Forest":
#                 return RandomForestClassifier() if is_classification else RandomForestRegressor()
#             elif self.algorithm == "AdaBoost":
#                 return AdaBoostClassifier() if is_classification else AdaBoostRegressor()
#             else:
#                 raise ValueError("Invalid algorithm specified.")
#         except Exception as e:
#             print(f"Error selecting model: {e}")
#             return None

#     def train_model(self):
#         """
#         Train the selected machine learning model.
#         """
#         try:
#             if self.model is not None:
#                 self.model.fit(self.X_train, self.y_train)
#                 print("Model training completed.")
#             else:
#                 print("Model is not initialized.")
#         except Exception as e:
#             print(f"Error training the model: {e}")

#     def evaluate_model(self):
#         """
#         Evaluate the trained model on the test set.
#         """
#         try:
#             if self.model is not None:
#                 score = self.model.score(self.X_test, self.y_test)
#                 print(f"Model evaluation score: {score}")
#             else:
#                 print("Model is not initialized.")
#         except Exception as e:
#             print(f"Error evaluating the model: {e}")

#     def predict(self, new_data):
#         """
#         Predict using the trained model on new data.
#         """
#         try:
#             if self.model is not None:
#                 new_data_preprocessed = self._preprocess_data(new_data)
#                 predictions = self.model.predict(new_data_preprocessed)
                
#                 # Map numerical predictions back to product names
#                 predicted_product_names = [self.product_names[int(pred)] for pred in predictions]
#                 return predicted_product_names
#             else:
#                 print("Model is not initialized.")
#                 return None
#         except Exception as e:
#             print(f"Error making predictions: {e}")
#             return None
