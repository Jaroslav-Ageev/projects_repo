import pandas as pd
import plotly.express as px

"""EDA Analysis of loans..."""

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)

df = pd.read_csv('PPP Data 150k plus 080820.csv')

#Examining Dimensions, Datatypes and Missing Values.

print("Table Shape")
print(df.shape)
print("\n")

print("Table Info")
df.info()
print("\n")

print("Columns with null values")
print("\n")
print(df.isna().sum())
print("\n")


#Checking Statistical properties
print("Mean and Percentiles")
print("\n")
print(df.describe())
print("\n")

print("Most Frequent values")
print("\n")
print(df[['State', 'City', 'Lender', 'BusinessType']].mode())
print("\n")

print("Top 5 banks by number of loans provided")
print("\n")
print(df['Lender'].value_counts()[:5])
print("\n")

print("Count of Loan ranges")
print("\n")
print(df['LoanRange'].value_counts())
print("\n")

print("Total number of banks participating in lending funds")
print(df['Lender'].unique().shape)
print("\n")

#Adding 2 columns for LoanRange: Upper bound and Lower bound

lower_bounds = {'a $5-10 million': 5000000,
               'b $2-5 million': 2000000,
               'c $1-2 million': 1000000,
               'd $350,000-1 million': 350000,
               'e $150,000-350,000': 150000}

upper_bounds = {'a $5-10 million': 10000000,
               'b $2-5 million': 5000000,
               'c $1-2 million': 2000000,
               'd $350,000-1 million': 1000000,
               'e $150,000-350,000': 350000}

df['LoanLowerBound'] = df['LoanRange'].replace(lower_bounds)
df['LoanUpperBound'] = df['LoanRange'].replace(upper_bounds)






