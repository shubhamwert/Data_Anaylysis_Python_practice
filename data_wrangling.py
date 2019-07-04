import pandas as pd
import matplotlib as plt
import numpy as np
from matplotlib import pyplot

filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df=pd.read_csv(filename,names=headers)
print(df.describe())

#getting missing data out

#first lets get idea how much data is missing

df.replace("?",np.nan,inplace=True)
print(df.head())

#better idea lets count them
print("\n\n\n\n\n\ncounting nans or missing data\n\n\n\n\n")
missingData=df.isnull()
for column in missingData.columns.values.tolist():
    print(column)
    print (missingData[column].value_counts())
    print("") 
#dealing with data
avg_norm_loss=df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)
df["bore"].replace(np.nan, avg_bore, inplace=True)
df["stroke"].replace(np.nan,df["bore"].astype('float').mean(axis=0),inplace=True)
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
df['num-of-doors'].value_counts().idxmax()
df["num-of-doors"].replace(np.nan, "four", inplace=True)
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

#correcting data formats
print("\n\n\n\n\n\n")
print(df.dtypes)

#casting into proper data_types
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

#standarizing

df['city-L/100km'] = 235/df["city-mpg"]
df["highway-L/100km"]=235/df["highway-mpg"]
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
print(df.head())

df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height']=df['height']/df['height'].max()


#binning and ploting
df["horsepower"]=df["horsepower"].astype(int, copy=True)
plt.pyplot.hist(df["horsepower"])
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


plt.pyplot.close()

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins

group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


df["horsepower-binned"].value_counts()

pyplot.bar(group_names, df["horsepower-binned"].value_counts())


plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

plt.pyplot.close()
a = (0,1,2)

plt.pyplot.hist(df["horsepower"], bins = 3)

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


#adding dummy variables
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
dummy_variable_1.head()
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("fuel-type", axis = 1, inplace=True)

dummy_variable_2=pd.get_dummies(df['aspiration'])
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop('aspiration', axis = 1, inplace=True)

print(df.columns)
print(df.head)


#saving the file

df.to_csv('data_wrangling_save.csv')