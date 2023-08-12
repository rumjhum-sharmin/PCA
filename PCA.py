# Reference: https://statisticsglobe.com/principal-component-analysis-python

# Step 1: Libraries and Data Preparation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
# Load sample csv file to pandas data_frame
df = pd.read_csv("F:/python/PCA/sample.csv")
print(df.to_string())

# Step 2: Data Standardization
scaler = StandardScaler()
scaler.fit(df)
mydata_scaled = scaler.transform(df)
print(mydata_scaled)

dataframe_scaled = pd.DataFrame(data=mydata_scaled, 
                                columns=df.columns)

print(dataframe_scaled.head(6))

# Step 3: Ideal Number of Components
#n-components means :  Total no of variables in tour dataset
pca = PCA(n_components=2) 
pca.fit_transform(mydata_scaled)
prop_var = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_

PC_numbers = np.arange(pca.n_components_) + 1
 
plt.plot(PC_numbers, 
         prop_var, 
         'ro-')
plt.title('Figure 1: Scree Plot', fontsize=8)
plt.ylabel('Proportion of Variance', fontsize=8)
plt.show()
#Step 4: Principal Component Calculation and Result Interpretation
pca = PCA(n_components=1)
PC = pca.fit_transform(mydata_scaled)
pca_mydata = pd.DataFrame(data = PC, 
                            columns = ['PC1'])

     