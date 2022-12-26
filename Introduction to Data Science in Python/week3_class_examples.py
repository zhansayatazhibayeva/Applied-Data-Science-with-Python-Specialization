#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

staff_df = pd.DataFrame([{'Name' : 'Ayazhan' , 'Tutor' : '10 A'},
                        {'Name' : 'Zhansaya' , 'Tutor' :'11 A'},
                        {'Name' : 'Ayaulim', 'Tutor' : '11 B'}])
staff_df = staff_df.set_index('Name')


# In[6]:


student_df = pd.DataFrame([{'Name' : 'Ayazhan' , 'UNI' : 'AIU'},
                          {'Name' : 'Zhansaya' , 'UNI' :'ENU'},
                          {'Name' : 'Bibka' ,'UNI' : 'ENU'}])
student_df = student_df.set_index('Name')
pd.merge(staff_df,student_df,how = 'outer', left_index = True,right_index = True)


# In[7]:


pd.merge(staff_df,student_df,how = 'inner',left_index = True,right_index = True)


# In[8]:


pd.merge(staff_df,student_df,how = 'left',left_index = True,right_index = True)


# In[10]:


pd.merge(staff_df,student_df,how = 'right',left_index = True,right_index = True)


# In[13]:


staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
pd.merge(staff_df, student_df, how = 'right',on = 'Name')


# In[14]:


staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 
                          'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 
                          'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 
                          'Location': 'Washington Avenue'}])
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 
                            'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 
                            'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 
                            'Location': '512 Wilson Crescent'}])

pd.merge(staff_df,student_df,how = 'right', on = 'Name')


# In[15]:


staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 
                          'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 
                          'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 
                          'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 
                            'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 
                            'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 
                            'School': 'Engineering'}]) 
pd.merge(staff_df,student_df,how = 'inner', on = ['First Name' , 'Last Name'])


# In[7]:


import numpy as np
import pandas as pd
import timeit

df = pd.read_csv('datasets/census.csv')
df.head()


# In[9]:


(df.where(df['SUMLEV']==50)
    .dropna()
    .set_index(['STNAME','CTYNAME'])
    .rename(columns = {'ESTIMATESBASE2010': 'Estimates Base 2010'}))
    


# In[10]:


def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    row['max']  = np.max(data)
    row['min']  = np.min(data)
    return row
df.apply(min_max,axis = 'columns')


# In[11]:


def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min' : np.min(data),'max' : np.max(data)})
df.apply(min_max,axis = 1).head()


# In[12]:


rows = ['POPESTIMATE2010', 'POPESTIMATE2011', 'POPESTIMATE2012', 'POPESTIMATE2013','POPESTIMATE2014', 
        'POPESTIMATE2015']
df.apply(lambda x : np.max(x[rows]),axis = 1).head()


# In[13]:


def get_state_region(x):
    northeast = ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 
                 'Rhode Island','Vermont','New York','New Jersey','Pennsylvania']
    midwest = ['Illinois','Indiana','Michigan','Ohio','Wisconsin','Iowa',
               'Kansas','Minnesota','Missouri','Nebraska','North Dakota',
               'South Dakota']
    south = ['Delaware','Florida','Georgia','Maryland','North Carolina',
             'South Carolina','Virginia','District of Columbia','West Virginia',
             'Alabama','Kentucky','Mississippi','Tennessee','Arkansas',
             'Louisiana','Oklahoma','Texas']
    west = ['Arizona','Colorado','Idaho','Montana','Nevada','New Mexico','Utah',
            'Wyoming','Alaska','California','Hawaii','Oregon','Washington']
    
    if x in northeast:
        return "Northeast"
    elif x in midwest:
        return "Midwest"
    elif x in south:
        return "South"
    else:
        return "West"


# In[14]:


df['state_region'] = df['STNAME'].apply(lambda x : get_state_region(x))


# In[15]:


df


# In[16]:


df[['STNAME','state_region']].head()


# In[17]:


df = df[df['SUMLEV'] == 50]
df.head()


# In[20]:


get_ipython().run_cell_magic('timeit', '-n 3', "\nfor state in df['STNAME'].unique():\n    # We'll just calculate the average using numpy for this particular state\n    avg = np.average(df.where(df['STNAME']==state).dropna()['CENSUS2010POP'])\n    # And we'll print it to the screen\n    print('Counties in state ' + state + \n          ' have an average population of ' + str(avg))")


# In[21]:


get_ipython().run_cell_magic('timeit', '-n 3', "for group,frame in df.groupby('STNAME'):\n    avg = np.average(frame['CENSUS2010POP'])\n    print('Counties in state ' + group + \n          ' have an average population of ' + str(avg))\n    ")


# In[22]:


df = pd.read_csv('datasets/listings.csv')
df.head()


# In[23]:


df = df.set_index(['cancellation_policy','review_scores_value'])
df.head()


# In[24]:


for group,frame in df.groupby(level=(0,1)):
    print(group)


# In[25]:


def grouping_by(item):
    if item[1]==10:
        return (item[0],'10.0')
    else:
        return (item[0],'not 10.0')
    
for group,frame in df.groupby(by = grouping_by):
    print(group)


# In[26]:


df = df.reset_index()
df.groupby('cancellation_policy').agg({'review_scores_value':np.nanmean})


# In[27]:


df.groupby('cancellation_policy').agg({'review_scores_value':(np.nanmean,np.nanstd),
                                      'reviews_per_month' : np.nanmean})


# In[28]:


df = pd.read_csv('datasets/cwurData.csv')
df.head()


# In[30]:


df['rank_level'] = 0
df.head()


# In[34]:


def name_rank(x):
    if x < 100:
        return 'first tier'
    elif x<200:
        return 'second tier'
    elif x<300:
        return 'third tier'
    else:
        return 'top tier'
#df['state_region'] = df['STNAME'].apply(lambda x : get_state_region(x))
df['rank_level'] = df['world_rank'].apply(lambda x : name_rank(x))



# In[35]:


df.pivot_table(values = 'score', index = 'country', columns = 'rank_level', aggfunc = [np.mean, np.max]).head()


# In[ ]:




