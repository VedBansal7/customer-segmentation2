# import tkinter as tk  # Importing tkinter for GUI creation
# from tkinter import ttk, filedialog, messagebox  # Importing additional tkinter widgets and dialogs
# from ml_engine import MLEngine  # Importing the MLEngine class
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import numpy as np

# class MLWindow:
#     def __init__(self, master):
#         """
#         Initialize the MLWindow with GUI components for selecting an algorithm,
#         uploading a CSV file.
#         """
#         self.master = master
#         self.window = tk.Toplevel(master)
#         self.window.title("Upload Data")
#         self.window.geometry("900x700")  # Increase window size to accommodate charts

#         self.csv_path = tk.StringVar()
#         self.target_column = "Price"  # Ensure this column exists in your dataset

#         # Creating a button to upload a CSV file
#         self.upload_button = ttk.Button(self.window, text="Upload CSV", command=self.upload_csv)
#         self.upload_button.pack(pady=10)

#         self.file_label = ttk.Label(self.window, textvariable=self.csv_path)
#         self.file_label.pack(pady=5)

#         # Creating a button to run the segmentation process
#         self.run_button = ttk.Button(self.window, text="Run Segmentation", command=self.run_segmentation)
#         self.run_button.pack(pady=10)

#         # Figure for plotting charts
#         self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 subplot for bar charts
#         self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
#         self.canvas.get_tk_widget().pack(pady=20)

#     def upload_csv(self):
#         """
#         Open a file dialog to select a CSV file and set the file path.
#         """
#         try:
#             file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
#             if file_path:
#                 self.csv_path.set(file_path)
#         except Exception as e:
#             messagebox.showerror("File Upload Error", f"Error uploading file: {e}")

#     def run_segmentation(self):
#         """
#         Run the segmentation using all four algorithms and plot the results.
#         """
#         try:
#             if not self.csv_path.get():
#                 messagebox.showwarning("No File", "Please upload a CSV file first!")
#                 return

#             # Ensure the correct target column is available in the dataset
#             algorithms = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost"]
#             axes = self.axes.flatten()  # Flatten the axes array for easier access

#             for i, algorithm in enumerate(algorithms):
#                 ml_engine = MLEngine(algorithm, self.csv_path.get(), self.target_column)
#                 result = ml_engine.run()

#                 # Plot a bar chart for each algorithm
#                 self.plot_bar_chart(result, algorithm, axes[i])

#             self.canvas.draw()  # Update the canvas with the new charts
#         except Exception as e:
#             messagebox.showerror("Segmentation Error", f"Error running segmentation: {e}")

#     def plot_bar_chart(self, result, algorithm, ax):
#         """
#         Plot bar chart for each algorithm based on the result.
#         The bar chart will show the number of predictions for each product category.
#         """
#         predictions = result['Predictions']
#         unique, counts = np.unique(predictions, return_counts=True)

#         ax.clear()
#         ax.bar(unique, counts, color='skyblue')
#         ax.set_title(f'{algorithm} - Predictions')
#         ax.set_xlabel('Product')
#         ax.set_ylabel('Count')
#         ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
#     # Add this method to the MLEngine class

# # Add this method to the MLEngine class

#     def run(self):
#       """
#        Train the model, evaluate it, and return the results.
#       """
#       try:
#            self.train_model()
#            self.evaluate_model()
        
#         # Make predictions on the test set
#            predictions = self.model.predict(self.X_test)
        
#            return {'Predictions': predictions}  # You can also return additional metrics if needed
#       except Exception as e:
#         print(f"Error running model: {e}")
#         return None
















import tkinter as tk  # Importing tkinter for GUI creation
from tkinter import ttk, filedialog, messagebox  # Importing additional tkinter widgets and dialogs
from ml_engine import MLEngine  # Importing the MLEngine class
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class MLWindow:
    def __init__(self, master):
        """
        Initialize the MLWindow with GUI components for selecting an algorithm,
        uploading a CSV file.
        """
        self.master = master
        self.window = tk.Toplevel(master)
        self.window.title("Upload Data")
        self.window.geometry("400x300")  # Initial window size

        self.csv_path = tk.StringVar()
        self.target_column = "Price"  # Ensure this column exists in your dataset

        # Creating a button to upload a CSV file
        self.upload_button = ttk.Button(self.window, text="Upload CSV", command=self.upload_csv)
        self.upload_button.pack(pady=10)

        self.file_label = ttk.Label(self.window, textvariable=self.csv_path)
        self.file_label.pack(pady=5)

        # Create a dropdown menu for selecting algorithms
        self.algorithm_var = tk.StringVar()
        self.algorithm_dropdown = ttk.Combobox(self.window, textvariable=self.algorithm_var)
        self.algorithm_dropdown['values'] = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost"]
        self.algorithm_dropdown.set("Select Algorithm")  # Default text
        self.algorithm_dropdown.pack(pady=10)

        # Creating a button to run the segmentation process
        self.run_button = ttk.Button(self.window, text="Run Segmentation", command=self.run_segmentation)
        self.run_button.pack(pady=10)

    def upload_csv(self):
        """
        Open a file dialog to select a CSV file and set the file path.
        """
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.csv_path.set(file_path)
        except Exception as e:
            messagebox.showerror("File Upload Error", f"Error uploading file: {e}")

    def run_segmentation(self):
        """
        Run the segmentation using the selected algorithm and plot the results.
        """
        try:
            if not self.csv_path.get():
                messagebox.showwarning("No File", "Please upload a CSV file first!")
                return

            algorithm = self.algorithm_var.get()
            if algorithm == "Select Algorithm":
                messagebox.showwarning("Select Algorithm", "Please select an algorithm!")
                return
            
            ml_engine = MLEngine(algorithm, self.csv_path.get(), self.target_column)
            result = ml_engine.run()

            # Open a new window for displaying the graph and predictions
            self.display_results(result, algorithm)
        except Exception as e:
            messagebox.showerror("Segmentation Error", f"Error running segmentation: {e}")

    def display_results(self, result, algorithm):
        """
        Display results in a new window with graph and predictions.
        """
        result_window = tk.Toplevel(self.window)
        result_window.title(f"{algorithm} Results")
        result_window.geometry("800x600")

        # Create figure for plotting chart
        fig, ax = plt.subplots(figsize=(8, 5))
        
        predictions = result['Predictions']
        unique, counts = np.unique(predictions, return_counts=True)

        ax.bar(unique, counts, color='skyblue')
        ax.set_title(f'{algorithm} - Predictions')
        ax.set_xlabel('Product')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

        # Add the chart to the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.get_tk_widget().pack(pady=20)
        canvas.draw()

        # Show predictions in a text box
        predictions_text = tk.Text(result_window, wrap=tk.WORD, width=80, height=10)
        predictions_text.pack(pady=10)

        prediction_message = f"Predictions for {algorithm}:\n\n" + "\n".join(f"Product: {p}" for p in predictions)
        predictions_text.insert(tk.END, prediction_message)
        predictions_text.config(state=tk.DISABLED)  # Make text box read-only




















# import tkinter as tk  # Importing tkinter for GUI creation
# from tkinter import ttk, filedialog, messagebox  # Importing additional tkinter widgets and dialogs
# from ml_engine import MLEngine  # Importing the MLEngine class
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import numpy as np

# class MLWindow:
#     def __init__(self, master):
#         """
#         Initialize the MLWindow with GUI components for selecting an algorithm,
#         uploading a CSV file.
#         """
#         self.master = master
#         self.window = tk.Toplevel(master)
#         self.window.title("Upload Data")
#         self.window.geometry("900x700")  # Increase window size to accommodate charts

#         self.csv_path = tk.StringVar()
#         self.target_column = "Price"  # Ensure this column exists in your dataset

#         # Creating a button to upload a CSV file
#         self.upload_button = ttk.Button(self.window, text="Upload CSV", command=self.upload_csv)
#         self.upload_button.pack(pady=10)

#         self.file_label = ttk.Label(self.window, textvariable=self.csv_path)
#         self.file_label.pack(pady=5)

#         # Creating a dropdown menu to select algorithms
#         self.algorithm_var = tk.StringVar(value="Logistic Regression")
#         self.algorithm_dropdown = ttk.Combobox(self.window, textvariable=self.algorithm_var, state='readonly')
#         self.algorithm_dropdown['values'] = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost"]
#         self.algorithm_dropdown.pack(pady=10)

#         # Creating a button to run the segmentation process
#         self.run_button = ttk.Button(self.window, text="Run Segmentation", command=self.run_segmentation)
#         self.run_button.pack(pady=10)

#     def upload_csv(self):
#         """
#         Open a file dialog to select a CSV file and set the file path.
#         """
#         try:
#             file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
#             if file_path:
#                 self.csv_path.set(file_path)
#         except Exception as e:
#             messagebox.showerror("File Upload Error", f"Error uploading file: {e}")

#     def run_segmentation(self):
#         """
#         Run the segmentation using the selected algorithm and plot the results.
#         """
#         try:
#             if not self.csv_path.get():
#                 messagebox.showwarning("No File", "Please upload a CSV file first!")
#                 return

#             algorithm = self.algorithm_var.get()
#             ml_engine = MLEngine(algorithm, self.csv_path.get(), self.target_column)

#             # Train the model
#             ml_engine.train_model()
#             # Run predictions on the test set
#             predictions = ml_engine.predict(ml_engine.X_test)

#             # Create a new window to display results
#             self.display_results(predictions, algorithm)
#         except Exception as e:
#             messagebox.showerror("Segmentation Error", f"Error running segmentation: {e}")

#     def display_results(self, predictions, algorithm):
#         """
#         Display results in a new window with graph and predictions.
#         """
#         result_window = tk.Toplevel(self.window)
#         result_window.title(f"{algorithm} Results")
#         result_window.geometry("800x600")

#         # Create figure for plotting chart
#         fig, ax = plt.subplots(figsize=(8, 5))
        
#         unique, counts = np.unique(predictions, return_counts=True)

#         ax.bar(unique, counts, color='skyblue')
#         ax.set_title(f'{algorithm} - Predictions')
#         ax.set_xlabel('Product')
#         ax.set_ylabel('Count')
#         ax.tick_params(axis='x', rotation=45)

#         # Add the chart to the Tkinter window
#         canvas = FigureCanvasTkAgg(fig, master=result_window)
#         canvas.get_tk_widget().pack(pady=20)
#         canvas.draw()

#         # Show predictions in a text box
#         predictions_text = tk.Text(result_window, wrap=tk.WORD, width=80, height=10)
#         predictions_text.pack(pady=10)

#         # Modify how predictions are displayed
#         prediction_message = f"Predictions for {algorithm}:\n\n" + "\n".join(f"Predicted Product: {name}" for name in predictions)
        
#         predictions_text.insert(tk.END, prediction_message)
#         predictions_text.config(state=tk.DISABLED)  # Make text box read-only

# if __name__ == "__main__":
#     root = tk.Tk()  # Create the main Tkinter window
#     app = MLWindow(root)  # Create an instance of MLWindow
#     root.mainloop()  # Run the application
