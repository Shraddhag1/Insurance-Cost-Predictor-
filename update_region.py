import pandas as pd


file_path = r'C:\Users\HP\OneDrive\Desktop\INSURANCE\Python\insurance.csv'


df = pd.read_csv(file_path)


indian_cities = ['Mumbai', 'Delhi', 'Pune', 'Bangalore']
df['region'] = [indian_cities[i % len(indian_cities)] for i in range(len(df))]


df.to_csv(file_path, index=False)

print("Region column updated with Indian cities.")