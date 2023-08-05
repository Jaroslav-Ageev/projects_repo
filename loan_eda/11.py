import plotly.express as px
import pandas as pd


df = pd.read_csv('PPP Data 150k plus 080820.csv')

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

fig=px.line(df, x="DateApproved", y="LoanLowerBound", color="State",
        template='simple_white')

fig.show()