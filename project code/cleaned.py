import numpy as np
import pandas as pd
import math

# Run this code to load the data
startups = pd.read_csv('startup-data.csv', index_col=0)
print(startups.shape)
startups.head()

startups.isnull().sum()

startups = startups.drop(['Unnamed: 6'], axis=1)

startups.isnull().sum()

print(startups['state_code.1']==startups['state_code'])
print(startups['id']==startups['object_id'])

startups = startups.drop(['state_code.1'], axis=1)
startups = startups.drop(['object_id'], axis=1)
start = ['c:']
end = ['']
startups['id'] = startups['id'].replace(start, end, regex=True)
startups.avg_participants = startups.avg_participants.round(4)

# startups['closed_at'] = startups['closed_at'].fillna('not closed')
startups['age_first_milestone_year'] = startups['age_first_milestone_year'].fillna(0)
startups['age_last_milestone_year'] = startups['age_last_milestone_year'].fillna(0)
startups.head()

#make a new column for years of the startup/how long it has been around
startups['closed_at'] = pd.to_datetime(startups['closed_at'])
startups['founded_at'] = pd.to_datetime(startups['founded_at'])

startups["years"] = (startups["closed_at"]-startups["founded_at"])
startups["years"]=round(startups.years/np.timedelta64(1,'Y'))

startups.head()
