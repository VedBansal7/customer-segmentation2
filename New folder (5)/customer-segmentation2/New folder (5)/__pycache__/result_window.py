import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

class ResultWindow:
    def __init__(self, master, results):
        self.master = master
        self.results = results

    def show(self):
        app = QApplication(sys.argv)
        window = QWidget()
        window.setWindowTitle("Results")

        layout = QVBoxLayout()

        labels = "\n".join([f"Algorithm: {result['Algorithm']} - Accuracy: {result['Accuracy']:.2f}" for result in self.results])
        label = QLabel(labels)
        layout.addWidget(label)

        chart_button = QPushButton("Show Chart")
        chart_button.clicked.connect(self.display_comparison_chart)
        layout.addWidget(chart_button)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        window.setLayout(layout)
        window.show()
        sys.exit(app.exec_())

    def display_comparison_chart(self):
        self.figure.clear()

        ax = self.figure.add_subplot(111)
        algorithms = [result['Algorithm'] for result in self.results]
        accuracies = [result['Accuracy'] for result in self.results]

        ax.bar(algorithms, accuracies)
        ax.set_title('Algorithm Accuracy Comparison')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Accuracy')

        # Invert the y-axis to have the bars start from the bottom
        ax.invert_yaxis()

        self.canvas.draw()








# import sys
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QRadioButton, QButtonGroup
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt
# import numpy as np

# class ResultWindow:
#     def __init__(self, master, results):
#         self.master = master
#         self.results = results

#     def show(self):
#         app = QApplication(sys.argv)
#         window = QWidget()
#         window.setWindowTitle("Results")

#         layout = QVBoxLayout()

#         labels = "\n".join([f"Algorithm: {result['Algorithm']} - Accuracy: {result['Accuracy']:.2f}" for result in self.results])
#         label = QLabel(labels)
#         layout.addWidget(label)

#         # Radio buttons for selecting chart type
#         self.chart_type_group = QButtonGroup(window)
#         bar_chart_radio = QRadioButton("Bar Chart")
#         pie_chart_radio = QRadioButton("Pie Chart")
#         line_chart_radio = QRadioButton("Line Chart")
        
#         bar_chart_radio.setChecked(True)  # Default to bar chart

#         self.chart_type_group.addButton(bar_chart_radio)
#         self.chart_type_group.addButton(pie_chart_radio)
#         self.chart_type_group.addButton(line_chart_radio)
        
#         layout.addWidget(bar_chart_radio)
#         layout.addWidget(pie_chart_radio)
#         layout.addWidget(line_chart_radio)

#         chart_button = QPushButton("Show Chart")
#         chart_button.clicked.connect(self.display_comparison_chart)
#         layout.addWidget(chart_button)

#         self.figure = plt.figure()
#         self.canvas = FigureCanvas(self.figure)
#         layout.addWidget(self.canvas)

#         window.setLayout(layout)
#         window.show()
#         sys.exit(app.exec_())

#     def display_comparison_chart(self):
#         self.figure.clear()

#         ax = self.figure.add_subplot(111)
#         algorithms = [result['Algorithm'] for result in self.results]
#         accuracies = [result['Accuracy'] for result in self.results]
#         colors = plt.cm.tab20.colors  # Use a color map for different colors

#         selected_chart_type = self.chart_type_group.checkedButton().text()

#         if selected_chart_type == "Bar Chart":
#             ax.bar(algorithms, accuracies, color=colors[:len(algorithms)])
#             ax.set_title('Algorithm Accuracy Comparison')
#             ax.set_xlabel('Algorithm')
#             ax.set_ylabel('Accuracy')
#             ax.invert_yaxis()

#         elif selected_chart_type == "Pie Chart":
#             ax.pie(accuracies, labels=algorithms, autopct='%1.1f%%', colors=colors[:len(accuracies)])
#             ax.set_title('Algorithm Accuracy Distribution')

#         elif selected_chart_type == "Line Chart":
#             ax.plot(algorithms, accuracies, marker='o', color='b')  # Single color for line chart
#             ax.set_title('Algorithm Accuracy Trend')
#             ax.set_xlabel('Algorithm')
#             ax.set_ylabel('Accuracy')
#             ax.set_xticks(range(len(algorithms)))
#             ax.set_xticklabels(algorithms, rotation=45, ha='right')

#         self.figure.tight_layout()  # Ensure layout is adjusted
#         self.canvas.draw()
