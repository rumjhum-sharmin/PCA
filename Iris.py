# open 'powershell' in computer, write 'pip install 'pandas'(library), 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# STEP 1: LOAD THE IRIS DATA SET
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = "F:/python/PCA/iris.data"
# load dataset into Pandas DataFrame
#if you have excell file then pd.read_csv("F:/python/PCA/iris.data")(location in your computer),this dataset has no column in csv file defined, thats why they defined like this according column)
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
print(df.head(5))

# #STEP 2: STANDARDIZE THE DATA
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Separating out the features
x = df.loc[:, features].values
print(x[:5])
# Separating out the target
y = df.loc[:,['target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)
print(x[:5])

# STEP 3: PCA PROJECTION TO 2D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(finalDf.head(5))

# STEP 4: VISUALIZE 2D PROJECTION
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
fig.savefig('Irish.png')
fig.show()
