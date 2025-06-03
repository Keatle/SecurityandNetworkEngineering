"""
Author : NTSIE KC 
Student No. : EDUV9197834
Question 2 
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 2.1 Read CSV file
file_path = Path(__file__).parent / "AmazonPrime.csv"
df = pd.read_csv(file_path)


summary = df.describe()

print("Summary Statistics:\n", summary)

# 2.2 Add 'Profitable' column (WorldGross > Budget)

df['Profitable'] = df['WorldGross'] > df['Budget']

# 2.3 Subset data with only 8 specified columns

subset_columns = ['Movie', 'RottenTomatoes', 'AudienceScore', 'Genre', 'WorldGross', 'Budget', 'Profitability', 'Profitable']
subset = df[subset_columns]
print("\nSubset of Data:\n", subset)

# 2.4 Filter films where WorldGross > 100

filtered = df[df['WorldGross'] > 100]
print("\nFilms with World Gross > 100:\n", filtered[['Movie', 'WorldGross']])

# 2.6    Plot distributions of AudienceScore and RottenTomatoes 

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['AudienceScore'], bins=10, kde=True, color='skyblue')
plt.title('Audience Score Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df['RottenTomatoes'], bins=10, kde=True, color='salmon')
plt.title('Rotten Tomatoes Score Distribution')

plt.tight_layout()
plt.show()


# 2.7 Use plot to analyse AudienceScore vs RottenTomatoes 

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='RottenTomatoes', y='AudienceScore', hue='Genre')
plt.title('Audience Score vs Rotten Tomatoes')
plt.xlabel('Rotten Tomatoes')
plt.ylabel('Audience Score')
plt.show()

# 2.8 Ave AudienceScore for each Genre
avg_scores = df.groupby('Genre')['AudienceScore'].mean().reset_index()
print("\nAverage Audience Score by Genre:\n", avg_scores)
