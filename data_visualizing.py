import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('telescope_data.csv')
for col in df.columns[1:-1]:
    plt.hist(df[df['class']=='g'][col],alpha=0.9,color='blue',label='gamma',density=True) #density for normalizing 
    plt.hist(df[df['class']=='h'][col],alpha=0.7,color='red',label='hedron',density=True) #desnity for normalizing
    plt.title(f'Data for {col}')
    plt.xlabel("Label")
    plt.ylabel('Probability')
    plt.legend()
    plt.show()