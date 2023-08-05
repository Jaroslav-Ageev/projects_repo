import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('PPP Data 150k plus 080820.csv')

msno.matrix(df)
plt.savefig('missingvalues.png', facecolor='w', dpi=300)