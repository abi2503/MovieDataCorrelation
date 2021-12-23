import os
import numpy as np
import pandas as pd
os.chdir("/Users/abhisheksuresh/Desktop")
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from matplotlib.pyplot import figure
#%matplotlib inline
#matplotlib.rcparams["figure.figsize"]=(12,8)#size of figures
movie=pd.read_csv("Kaggle_Case_Study/movies.csv")
print(np.sum(movie["runtime"].isnull()))
print(movie.describe())

print(movie["runtime"].isnull().sum())

print(movie["runtime"].sum(),movie["runtime"].count())

#function to check if there is any null values in data
for col in movie.columns:
    pct_missing_values=np.mean(movie[col].isnull())
    print(col,"",np.sum(movie[col].isnull()))
    #print(col," has ",pct_missing_values," % missing values")
    
#drop all null values

movie.dropna(inplace=True)#drops rows with null values

#function to check if there is any null values in data
for col in movie.columns:
    pct_missing_values=np.mean(movie[col].isnull())
    print(col,"",np.sum(movie[col].isnull()))
    #print(col," has ",pct_missing_values," % missing values")


movie['budget']=movie['budget'].astype('int64')
movie['gross']=movie['gross'].astype('int64')
# movie['released_correct']=movie['released'].astype(str)

# test=movie.loc[:,['released']]

# print(test.info)

print(movie['year'].value_counts())
print(movie.sort_values(by=['gross'],ascending=False,inplace=True))#sort the values
print(movie)

#drop duplicates 
#movie['company']=e['company'].drop_duplicates()

#visualisation


plt.scatter(x=movie['budget'],y=movie['gross'])
plt.title("Budget vs Gross")
plt.xlabel("Budget")
plt.ylabel("Gross")
plt.show()


#Plot budget vs gross using seaborn

sns.regplot(x="budget",y="gross",data=movie,scatter_kws={"color":"red"},line_kws={"color":"blue"})
plt.show()

#check correlation

correlation_matrix=movie.corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix,annot=True)
plt.title("Correlation for Numeric Values")
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")
plt.show()


#find correlation for all columns
movie_numerised=movie
for col in movie_numerised.columns:
    if(movie_numerised[col].dtypes==object):
        movie_numerised[col]=movie_numerised[col].astype("category")#convert non-numeric data into categorical
        movie_numerised[col]=movie_numerised[col].cat.codes
 
corr_matrix=movie_numerised.corr()
corr_pairs=corr_matrix.unstack()#gets the pairs
sns.heatmap(corr_matrix,annot=True)
plt.title("Correlation ")
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")
plt.show()
sorted_pairs=corr_pairs.sort_values()
high_corr=sorted_pairs[(sorted_pairs)>0.5]
print(high_corr)