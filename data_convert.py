# In convert_data.py
import numpy as np

try:
    # Load the data from your text file
    prices_array = np.loadtxt('prices.txt')

    # The data needs to be transposed to match the competition format (50 rows x 750 columns)
    prices_transposed = prices_array.T 

    # Save the array in the efficient .npy format
    np.save('prices.npy', prices_transposed)

    print("Successfully converted prices.txt to prices.npy")
    print(f"New shape of the data: {prices_transposed.shape}")

except FileNotFoundError:
    print("Error: prices.txt not found. Make sure it's in the same folder.")