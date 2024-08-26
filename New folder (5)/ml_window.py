import tkinter as tk  # Importing tkinter for GUI creation
from tkinter import ttk, filedialog, messagebox  # Importing additional tkinter widgets and dialogs
from ml_engine import MLEngine  # Importing the MLEngine class
from result_window import ResultWindow  # Importing the ResultWindow class

class MLWindow:
    def __init__(self, master):
        """
        Initialize the MLWindow with GUI components for selecting an algorithm,
        uploading a CSV file.
        """
        self.master = master
        self.window = tk.Toplevel(master)
        self.window.title("Upload Data")
        self.window.geometry("500x350")

        self.csv_path = tk.StringVar()
        self.target_column = "Price"  # Automatically setting the target column

        # Creating a button to upload a CSV file
        self.upload_button = ttk.Button(self.window, text="Upload CSV", command=self.upload_csv)
        self.upload_button.pack(pady=10)

        # Creating a label to display the file path
        self.file_label = ttk.Label(self.window, textvariable=self.csv_path)
        self.file_label.pack(pady=5)

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
        Run all segmentation algorithms and display the results.
        """
        try:
            if not self.csv_path.get():
                messagebox.showwarning("No File", "Please upload a CSV file first!")
                return

            algorithms = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost"]
            results = []

            for algorithm in algorithms:
                ml_engine = MLEngine(algorithm, self.csv_path.get(), self.target_column)
                result = ml_engine.run()
                results.append(result)

            result_window = ResultWindow(self.window, results)
            result_window.show()
        except Exception as e:
            messagebox.showerror("Segmentation Error", f"Error running segmentation: {e}")

    def show(self):
        """
        Display the MLWindow and wait for it to be closed.
        """
        try:
            self.window.grab_set()
            self.window.wait_window()
        except Exception as e:
            print(f"Error showing ML window: {e}")
