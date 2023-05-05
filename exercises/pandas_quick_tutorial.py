import numpy as np
import pandas as pd

# Task 1: Create a DataFrame
t1_columns = ['Eleanor','Chidi','Tahani','Jason']
t1_data = []
for i in range(3):
    data = np.random.randint(low=0, high=101, size=(4))
    t1_data.append(data)

t1_dataframe = pd.DataFrame(data=t1_data, columns=t1_columns)
print(t1_dataframe)
print(t1_dataframe['Eleanor'][0])

t1_dataframe['Janet'] = t1_dataframe['Tahani'] + t1_dataframe['Jason']
print(t1_dataframe)