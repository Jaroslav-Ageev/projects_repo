import pandas as pd
import matplotlib.pyplot as plt

"""Building a bar plot: Business type for loan consumption"""

df = pd.read_csv('PPP Data 150k plus 080820.csv')

plt.figure(figsize=(30, 20))

ax=df['BusinessType'].value_counts().plot.bar()

# Add title and axis names
plt.title('quantity of loans by Business type', fontsize=25)
plt.xlabel('Business type', fontsize=20)
ax.xaxis.set_label_coords(.8, -.1)
plt.ylabel('Q of loans', fontsize=20)
plt.xticks(rotation=45, ha="right")

plt.savefig('loans by Business Type.png', facecolor='w', dpi=300)
