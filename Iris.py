import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.style.use('ggplot')
 
# Load sample csv file to pandas data_frame
df = pd.read_csv("F:/python/PCA/Iris.csv")
print(df.to_string())
scaler = StandardScaler()
scaler.fit(df)
Irisdata_scaled = scaler.transform(df)
print(Irisdata_scaled)
dataframe_scaled = pd.DataFrame(data=Irisdata_scaled, 
                              columns=df.columns)

print(dataframe_scaled.head(6))
pca = PCA(n_components=4) 
pca.fit_transform(Irisdata_scaled)
prop_var = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_

PC_numbers = np.arange(pca.n_components_) + 1
 
plt.plot(PC_numbers, 
          prop_var, 
          'ro-')
plt.title('Figure 1: Scree Plot', fontsize=8)
plt.ylabel('Proportion of Variance', fontsize=8)
plt.show()
# pca = PCA(n_components=4) 
# pca.fit_transform(Irisdata_scaled)
# prop_var = pca.explained_variance_ratio_
# eigenvalues = pca.explained_variance_

# PC_numbers = np.arange(pca.n_components_) + 1
 
# plt.plot(PC_numbers, 
#          prop_var, 
#          'ro-')
# pca = PCA(n_components=2)
# PC = pca.fit_transform(Irisdata_scaled)
# pca_Irisdata = pd.DataFrame(data = PC, 
#                             columns = ['PC1','PC2'])