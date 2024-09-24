import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/devba/OneDrive/Desktop/online_retail_store.csv'
data = pd.read_csv(file_path)

# Calculate total sales per product
data['Total Sales'] = data['Price'] * data['Quantity']
product_sales = data.groupby('Product')['Total Sales'].sum()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(product_sales, labels=product_sales.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Product Sales Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.show()
