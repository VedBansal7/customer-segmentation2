# import tkinter as tk  # Importing tkinter for GUI creation
# from tkinter import ttk  # Importing themed tkinter widgets
# from ml_window import MLWindow  # Importing the MLWindow class

# class MainApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Customer Segmentation")
#         self.root.geometry("300x200")

#         # Creating a button to open the MLWindow
#         self.ml_button = ttk.Button(root, text="Start Segmentation", command=self.open_ml_window)
#         self.ml_button.pack(pady=20)

#     def open_ml_window(self):
#         try:
#             ml_window = MLWindow(self.root)
#             ml_window.show()
#         except Exception as e:
#             print(f"Error opening ML window: {e}")

# if __name__ == "__main__":
#     try:
#         root = tk.Tk()
#         app = MainApp(root)
#         root.mainloop()
#     except Exception as e:
#         print(f"Error in main application: {e}")





# main.py

import tkinter as tk  # Importing tkinter for GUI creation
from tkinter import ttk  # Importing themed tkinter widgets
from ml_window import MLWindow  # Importing the MLWindow class

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Segmentation")
        self.root.geometry("300x200")

        # Creating a button to open the MLWindow
        self.ml_button = ttk.Button(root, text="Start Segmentation", command=self.open_ml_window)
        self.ml_button.pack(pady=20)

    def open_ml_window(self):
        try:
            # Simply initialize MLWindow without the show method
            MLWindow(self.root)
        except Exception as e:
            print(f"Error opening ML window: {e}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = MainApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error in main application: {e}")
