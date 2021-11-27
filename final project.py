# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%

import time
start=time.perf_counter()

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%

def cleanDfLocation(row, colname):
  place = row[colname].strip()
  if('CA' in place or 'NV' in place or 'HI' in place): return 'West'
  if('WA' in place or 'OR' in place or 'ID' in place or 'MT' in place or 'WY' in place or 'AK' in place): return 'Northwest'
  if('UT' in place or 'AZ' in place or 'CO' in place or 'NM' in place or 'TX' in place or 'OK' in place): return 'Southwest'
  if('ND' in place or 'SD' in place or 'NE' in place or 'KS' in place or 'MN' in place or 'IA' in place): return 'Midwest'
  if('MO' in place or 'WI' in place or 'IL' in place or 'MI' in place or 'IN' in place or 'OH' in place): return 'Midwest'
  if('KY' in place): return 'Midwest'
  if('LA' in place or 'AR' in place or 'MS' in place or 'TN' in place or 'AL' in place or 'GA' in place): return 'Southeast'
  if('SC' in place or 'NC' in place or 'FL' in place): return 'Southeast'
  if('VA' in place or 'WV' in place or 'DC' in place or 'PA' in place or 'MD' in place or 'DE' in place): return 'Mid-Atlantic'
  if('NJ' in place or 'NY' in place): return 'Mid-Atlantic'
  if('VT' in place or 'NH' in place or 'CT' in place or 'RI' in place or 'MA' in place or 'ME' in place): return 'Northeast'
  else: return 'International'
  return np.nan

#%%
df = pd.read_csv("salary.csv")

#%%

df['location'] = df.apply(cleanDfLocation, colname='location', axis=1)

#%%

# drop NA 
df_nona = df.dropna(subset=["Education"]) # drop NA in Education
df_nona = df_nona.dropna(subset=["Race"]) # drop NA in race
df_nona = df_nona.dropna(subset=["gender"]) # drop NA in gender

#%%

# creat a new boolean variable as dependent variable for logistic regression
newdf = df_nona.iloc[:,[0,1,3,4,5,6,7,9,10,11,12,27,28]].copy() # pick up useful columns
newdf["stock"] = 0 # creat a new column
newdf["stock"][newdf.stockgrantvalue > 0] = 1 # person with granted stock is 1

#%%

# define a function to keep only year in timestamp
def cleanDfTime(row, colname):
  time = row[colname].strip()
  if('2021' in time): return '2021'
  if('2020' in time): return '2020'
  if('2019' in time): return '2019'
  if('2018' in time): return '2018'
  if('2017' in time): return '2017'
  else: return time
  
#%%
newdf['year'] = newdf.apply(cleanDfTime, colname='timestamp', axis=1)
newdf.head()
# Only observations in 2021 and 2020 has no NA in education, race and gender, so year only has two values(2020 and 2021)

# %%

print("Run time:",time.perf_counter()-start)
