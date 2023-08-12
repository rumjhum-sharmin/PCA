# Reference: https://statisticsglobe.com/principal-component-analysis-python

# Step 1: Libraries and Data Preparation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes
 
# Load sample csv file to pandas data_frame
# df = pd.read_csv("F:/python/PCA/sample.csv")
# print(df.to_string())

diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, 
                  columns=diabetes.feature_names) 
print(df.head(6))

# Step 2: Data Standardization
scaler = StandardScaler()
scaler.fit(df)
diabetes_scaled = scaler.transform(df)
print(diabetes_scaled)

dataframe_scaled = pd.DataFrame(data=diabetes_scaled, 
                                columns=df.columns)

print(dataframe_scaled.head(6))

# Step 3: Ideal Number of Components
#n-components means :  To see 10 principle component
pca = PCA(n_components=10) 
pca.fit_transform(diabetes_scaled)
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
pca = PCA(n_components=2)
PC = pca.fit_transform(diabetes_scaled)
pca_diabetes= pd.DataFrame(data = PC, 
                            columns = ['PC1','PC2'])
print(pca_diabetes.head(6))
# for biplot 
def biplot(score,coef,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coef.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,
                s=5, 
                color='orange')
 
    for i in range(n):
        plt.arrow(0, 0, coef[i,0], 
                  coef[i,1],color = 'purple',
                  alpha = 0.5)
        plt.text(coef[i,0]* 1.15, 
                 coef[i,1] * 1.15, 
                 labels[i], 
                 color = 'darkblue', 
                 ha = 'center', 
                 va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))    
    plt.title('Biplot of PCA')
    # plt.figure()
    return plt

# After defining our function, we just have to call it.

plt = biplot(PC, 
       np.transpose(pca.components_), 
       list(diabetes.feature_names))
plt.show()
