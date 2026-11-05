import os
from ucimlrepo import fetch_ucirepo
import pandas as pd

concrete = fetch_ucirepo(id=165) # 165 = this dataset’s UCI id 
X = concrete.data.features # pandas DataFrame, one column per input 
y = concrete.data.targets # pandas DataFrame with ONE column 
print(concrete.metadata) # dict: source, size, missing values, ... 
print(concrete.variables) # table: name, role, type, units of each variable

X = X.copy() 
X.columns = ["cement", "slag", "ash", "water", "superplastic", "coarse_agg", "fine_agg", "age"] 
y = y.iloc[:, 0].rename("strength") # one-column DataFrame-> Series

os.makedirs('PW4/Session A/data', exist_ok=True)
df = X.assign(strength=y)
df.to_csv('PW4/Session A/data/concrete.csv', index=False)
print("data are successfully installed and saved as data/concrete.csv")