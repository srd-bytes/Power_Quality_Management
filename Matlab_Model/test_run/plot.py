import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
df = pd.read_excel("bus_data_batch_0_1765647469.xlsx")   # change file name as needed

# Select the column to plot
column_name = "Bus3_Vb"           

# Plot the column
plt.figure(figsize=(10,5))
plt.plot(df[column_name])
plt.xlabel("Index")
plt.ylabel(column_name)
plt.title(f"{column_name} Plot")
plt.grid(True)
plt.show()
