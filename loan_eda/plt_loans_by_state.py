
import pandas as pd
import matplotlib.pyplot as plt

"""Building a plot: Quantity of loans given per state"""

df = pd.read_csv('PPP Data 150k plus 080820.csv')

plt.figure(figsize=(40, 20))
df['State'].hist(bins=150)

# Add title and axis names
plt.title('Loans by State', fontsize=50)
plt.xlabel('State', fontsize=30)
plt.ylabel('Q of loans', fontsize=30)

plt.savefig('loans by State.png', facecolor='w', dpi=300)



