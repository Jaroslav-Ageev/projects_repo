import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('PPP Data 150k plus 080820.csv')

sns.boxenplot(data=df, x='JobsReported', y='LoanRange', order=df['LoanRange'].\
              value_counts().index, color='white')

plt.savefig('jobs_to_loans.png')