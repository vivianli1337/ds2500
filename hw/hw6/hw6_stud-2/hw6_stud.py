#!/usr/bin/env python
# coding: utf-8

# # DS 2500 HW 6
# 
# Due: Fri Mar 24 @ 11:59PM
# 
# ### Submission Instructions
# Please submit both of the following to the corresponding [gradescope](https://www.gradescope.com/courses/478298) assignment:
# - this `.ipynb` file
#     - give a fresh `Kernel > Restart & Run All` just before uploading
# - a `.py` file consistent with your `.ipynb`
#     - `File > Download as ...`
# 
# ### Tips for success
# - Start early
# - Make use of [Piazza](https://course.ccs.neu.edu/ds2500/admin_piazza.html)
# - Make use of [Office Hours](https://course.ccs.neu.edu/ds2500/office_hours.html)
# - Remember that [Documentation / style counts for credit](https://course.ccs.neu.edu/ds2500/python_style.html)
# - [No student may view or share their ungraded homework with another](https://course.ccs.neu.edu/ds2500/syllabus.html#academic-integrity-and-conduct)
# 
# | part                                        |    |
# |:--------------------------------------------|---:|
# | Part 1.1: Car weight & power                | 15 |
# | Part 2: Polynomial Fitting                  | 35 |
# | Part 3: Clustering States by Driving Habits | 15 |
# | Part 4: PCA Iris                            | 15 |
# | total                                       | 80 |
# 
# This HW is a bit shorter than most to allow you time to work on your projects :)

# # Part 1.1: Car weight & power (15 points)
# 1. Given the data below, build and plot a `LinearRegression` model as shown immediately below:
# 
# <img src="https://i.ibb.co/W2X3BWb/horsepower-vs-weight.png" width=700>
# 
# Your output should be as aesthetically-good-looking and clear, but you needn't follow our color scheme / font sizes exactly.
# 

# In[1]:


import seaborn as sns

df_car = sns.load_dataset('mpg')
df_car.head()


# In[2]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[3]:


x_feat = 'horsepower'
y_feat = 'weight'

# drop the rows without a value
df_car.dropna(axis=0, how='any', inplace=True)

# extract values
x = df_car.loc[:, x_feat].values
y_true = df_car.loc[:, y_feat].values

# reshape
x = x.reshape(-1,1)

# fit and predict regression model
reg = LinearRegression()
reg.fit(x, y_true)
y_pred = reg.predict(x)

# plot it
plt.xlabel('horsepower')
plt.ylabel('weight')
plt.title('Relationship between Horsepower and Weight')
plt.scatter(x, y_true, label='observed', color='k')
plt.plot(x, y_pred, label='predict', color='r')
plt.legend()

sns.set(font_scale=1.4)

# plot_regression(x, y_true)
plt.gcf().set_size_inches(10, 7)


# # Part 1.2: Interpretting Regressions (20 points, 10 each)
# 
# Answer each of the questions below by
# - writing a few lines of relevant python code in one or two code cells
# - writing one or two clear, succinct sentence response in markdown just below
# 
# 1. Compute and interpret a quantification of how good the model in part 1.1 is.  How helpful is horsepower in explaining differences in the weight of a car for this particular set of cars?
# 1. One car has 50 more horsepower than another.  Using the model from part 1.1, whats our best guess as to how much heavier the more powerful car is?

# In[4]:


from sklearn.metrics import r2_score

# computing R2 from sklearn
r2 = r2_score(y_true=y_true, y_pred=y_pred)
r2


# ### <font color=lilac> Response: </font> 
# 
# The $R^2$ is 0.747, which displays that roughly 75% of the variations are explained by the model, and this number also indicates a moderate strong correlation between the x feature and y feature. Horsepower is related to the weight of the car; heavier cars have more horsepower. Based on the $R^2$, the model is helpful to a certain degree. However, there are some errors in the model aka the percentage of data (25%) that cannot be explain by the model.

# In[5]:


slope = reg.coef_[0]
weight = slope * 50
weight


# ### <font color=lilac> Response: </font> 
# The weight would increase by 19.1 per y value. Since the horsepower is increased by 50, the weight would increase by 953.91 lbs.

# # Part 2: Polynomial Fitting (35 points)
# 
# Identify the polynomial that `x, y` likely comes from.  Your response should be written out (may be done in markdown, not programmatically) as:
# 
# $$y = 1 + 2x + 3x^2$$
#     
# or similar.  Please round your coefficients to 2 decimal places so they're easily read.  Be sure to justify your chosen polynomial [degree](https://en.wikipedia.org/wiki/Degree_of_a_polynomial) with a graph and a sentence.
# 
# #### Hints:
# - How do I pick a polynomial degree?
#     - see "Preventing Overfitting" from day 17
#     - a `plt.plot()` of degree vs cross validated r2 might be a helpful
#     - “Everything should be made as simple as possible, but no simpler.” 
# - there is an order to our x variables, the earlier entries are often lower
#     - this feels like the chinstrap tragedy from day 13 ...
# - after cross validation I'll have a bunch of models, which do I report in my final polynomial estimate?
#     - none of these cross validated models are approrpiate

# In[6]:


import pickle

# loads arrays x and y from file
with open('xy_hw6.p', 'rb') as f:
    x, y = pickle.load(f)
    
# # having trouble with the pickle file?  use the csv as a backup
# df = pd.read_csv('xy_hw6.csv')
# x = df['x'].values
# y = df['y'].values

plt.scatter(x, y);


# In[7]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from copy import copy


# In[8]:


# find the best degree
r2_list = []

# set degree
for degree in range(1,12):
    # project x
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)
    
    # initalized kfold
    kfold = KFold(n_splits=10, shuffle=True)
    
    for train_idx, test_idx in kfold.split(x, y):
        # training data
        x_train = x[train_idx,:]
        y_train = y[train_idx]

        # testing data
        x_test = x[test_idx]

    # fit via linear regression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(x_poly, y)

    # predict y
    x_fine = np.linspace(x.min(), x.max(), 101).reshape(-1,1)
    x_fine_poly = poly.fit_transform(x_fine)
    y_pred_fine = reg.predict(x_fine_poly)

    # compute r2
    y_pred = reg.predict(x_poly)
    r2 = r2_score(y_true = y, y_pred = y_pred)
    r2_list.append(r2)

# plot
plt.plot(range(1,12),r2_list, label=f'r2={r2: .3f}', color='r', linewidth=3)
plt.title("r2 vs degree")
plt.xlabel("degree")
plt.ylabel("r2")
plt.legend()


# In[9]:


# plot with degree = 3
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

# fit via linear regression
reg = LinearRegression(fit_intercept=False)
reg.fit(x_poly, y)

# predict y
x_fine = np.linspace(x.min(), x.max(), 101).reshape(-1,1)
x_fine_poly = poly.fit_transform(x_fine)
y_pred_fine = reg.predict(x_fine_poly)

# compute r2
y_pred = reg.predict(x_poly)
r2 = r2_score(y_true = y, y_pred = y_pred)

# plot
plt.plot(x_fine, y_pred_fine, label=f'no scross val r2={r2: .3f}', 
         color='r', linewidth=3)
plt.scatter(x, y, label=f'observed', color='k')
plt.legend()

# find the coefficient
reg.coef_[0]


# ### <font color=lilac> Response: </font> 
# ## $y = 0.89 - 1.08x + 2.10x^2 - 0.52x^3$
# 
# Based on the r2 vs degree graph, it shows that degree = 1 and degree = 2 have a very low r2. However, r2 increase dramatically when degree is 3 and it remains constant after the degree increase. Thus, degree = 3 is the lowest polynomial degree with a high r2.
# 

# # Part 3: Clustering States by Driving Habits  (15 points)
# 
# Use K-Means clustering to cluster all the states into k sub-groups so that each sub-group has similar car crash statistics.  
# - Build a graph of how the mean distance from sample to centroid changes as k increases from two to seven.
# - Write one or two sentences which give the number of sub-groups best suited for this data (i.e. find the "elbow").  
#     - If no particular k seems much better than the others, characterize what about the graph leads you to this conclusion.
# 
# Hint:
# - is this raw data, without any preprocessing steps applied, appropriate for clustering?
#     - no

# In[10]:


df_car = sns.load_dataset('car_crashes')
df_car.head()


# In[11]:


from sklearn.cluster import KMeans

# extract x features
x_feat_list = df_car.columns[1:-1]
x = df_car.loc[:, x_feat_list].values

# normalize scale
x = x @ np.diag(1 / x.std(axis=0))

mean_d_dict = dict()
for n_clusters in range(2, 12):
    # fit kmeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(x)
    y = kmeans.predict(x)    
        
    # compute & store mean distance
    mean_d = -kmeans.score(x)
    mean_d_dict[n_clusters] = mean_d

plt.plot(mean_d_dict.keys(), mean_d_dict.values())
plt.xlabel('number of clusters')
plt.ylabel('mean dist^2 to centroid')
plt.title('Mean distance vs cluster size')


# ### <font color=lilac> Response: </font> 
# 
# Based on the mean distance vs cluster size elbow chart, there seems to be no good k-mean that has a good balance of simplicitiy while also ensuring centroids are close to their corresponding centroid. The line shows a slight change around n=4 and n=4, changing its direction track from a sleep slope to a slightly more gentle slope. However, it is not very sharp. As shown in the plots of clusters across all features below, there are no particular k that is better than the others. The points from each cluster are overlapping, and there are no definite groups.

# In[12]:


# Plots of Clusters across all features
from sklearn.cluster import KMeans

# get x from x in the previous cell
# extract x features
x = x

# perform clustering

for n_clusters in range(3, 6):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(x)
    y = kmeans.predict(x)

    # plot
    df_car['cluster'] = y
    plt.figure()
    sns.pairplot(data=df_car,hue='cluster', palette='Set2')

    plt.suptitle(f'{n_clusters} clusters of car w/ mean_d: {mean_d: 0.1e}', ha='center',
            va='baseline')
    plt.gcf().set_size_inches(15, 15)


# # Part 4: PCA Iris (15 points)
# Build the following "Principal Component Map" from all four features in the iris dataset below.  
# 
# <img src='https://i.ibb.co/2Ktt1Xm/iris-pca.png' width=800px>
# 
# You're welcome to submit a `matplotlib` scatter for full credit, but you may find the interactive `plotly` graph more fun to work with.  You can install plotly with `pip3 install plotly` or similar and build the necessary scatter plot with:
# ```python
# import plotly.express as px
# 
# fig = px.scatter(df_iris, x='pca0', y='pca1', hover_data=df_iris.columns, color='species')
# fig.show()
# 
# # if you want to export to html (not needed for HW at all, just good to know!)
# fig.write_html('iris_pca.html')
# ```

# In[13]:


df_iris = sns.load_dataset('iris')

df_iris.head()


# In[14]:


from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# In[15]:


x_feat_list = df_iris.columns[:-1]

# extract relevant x values
x = df_iris.loc[:, x_feat_list].values

# compress
pca = PCA(n_components=2, whiten=False)
x_compress = pca.fit_transform(x)

# add features back into dataframe (for plotting)
df_iris['pca0'] = x_compress[:, 0]
df_iris['pca1'] = x_compress[:, 1]

# scatter plot
fig = px.scatter(df_iris, x='pca0', y='pca1', hover_data=df_iris.columns, color='species')
fig.show()

# if you want to export to html (not needed for HW at all, just good to know!)
fig.write_html('iris_pca.html')

