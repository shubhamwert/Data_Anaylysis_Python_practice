import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from scipy import stats

path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


#learning corelation

print(df.corr())
#plotting it
sns.regplot(x="engine-size", y="price", data=df)
plt.pyplot.ylim(0,)

sns.regplot(x="peak-rpm", y="price", data=df)

df[['stroke','price']].corr()
sns.regplot(x="peak-rpm", y="price", data=df)

sns.boxplot(x="body-style", y="price", data=df)

df.describe(include=['object'])

df['drive-wheels'].value_counts().to_frame()

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts

drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)

df['drive-wheels'].unique()
df_group_one = df[['drive-wheels','body-style','price']]
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()

plt.pyplot.pcolor(grouped_pivot, cmap='RdBu')
plt.pyplot.colorbar()
plt.pyplot.show()


pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  