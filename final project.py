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
df = pd.read_csv("salary.csv")



#%%

numcomps=len(df.drop_duplicates(subset='company'))


# %%

#
# Below is the number of workers in different regions of the US
# 0=west, 1=northwest/Kanye's son, 2=soutwest, 3=midwest, 4=southeast, 5=mid-atlantic, 6=northeast
# I thought these values might be useful since Huang mentioned income vs region
# This can be repeated with different countries
# I saw "United Kingdonm", "Ireland", and "Russia" when I quickly looked through the list
# 

USworkers=np.zeros(7,int)

USworkers[0]=int(df['location'].str.contains('CA|NV|HI').sum())
USworkers[1]=int(df['location'].str.contains('WA|OR|ID|MT|WY|AK').sum())
USworkers[2]=int(df['location'].str.contains('UT|AZ|CO|NM|TX|OK').sum())
USworkers[3]=int(df['location'].str.contains('ND|SD|NE|KS|MN|IA|MO|WI|IL|MI|IN|OH|KY').sum())
USworkers[4]=int(df['location'].str.contains('LA|AR|MS|TN|AL|GA|SC|NC|FL').sum())
USworkers[5]=int(df['location'].str.contains('VA|WV|DC|PA|MD|DE|NJ|NY').sum())
USworkers[6]=int(df['location'].str.contains('VT|NH|CT|RI|MA|ME').sum())

print(sum(USworkers)/len(df))
print(USworkers)

# %%

print("Run time:",time.perf_counter()-start)
