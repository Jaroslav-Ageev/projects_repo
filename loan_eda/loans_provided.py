import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('PPP Data 150k plus 080820.csv')

plt.figure(figsize=(40, 20))

sns.countplot(data=df, y='LoanRange', order=df['LoanRange'].value_counts().\
                                            index, color='blue')
plt.title('Types of loans supplied', fontsize=30)
plt.xlabel('count', fontsize=25)
plt.ylabel('Loan Range', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)

plt.savefig('Types of loans supplied')