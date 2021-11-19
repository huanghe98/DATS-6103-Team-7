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

# %%

print("Run time:",time.perf_counter()-start)
