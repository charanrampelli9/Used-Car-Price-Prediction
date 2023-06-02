# %% [markdown]
# # Car Price Prediction
# 

# %%
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# %%
used_car_df=pd.read_csv("used_cars_train_data.csv")
used_car_df

# %%
used_car_df.isna().sum()

# %%
df2=used_car_df.copy()

# %%
df2.columns

# %%
df2.dtypes

# %%
boxplot = df2.boxplot(column=['Kilometers_Driven'])  

# %%
boxplot = df2.boxplot(column=['Seats'])  


# %%
boxplot = df2.boxplot(column=['Year'])  


# %%
df2['Owner_Type'].unique()

# %%

def label_function(val):
    return f'{val / 100 * len(df2):.0f}\n{val:.0f}%'
fig, (ax1) = plt.subplots(figsize=(10, 5))
df2.groupby('Owner_Type').size().plot(title='Owner Type',kind='pie', autopct=label_function, textprops={'fontsize': 20},
                                  ax=ax1)

# %%
df2['Year'].unique()

# %%


fig, (ax1) = plt.subplots(figsize=(10, 5))
df2.groupby('Year').size().plot(title='Year',kind='pie', textprops={'fontsize': 10},
                                  ax=ax1)

# %% [markdown]
# ### Data Cleaning ,Converting string to int

# %%
used_car_df.dtypes

# %% [markdown]
# ### Dividing into training and testing datasets

# %%
X_train, X_test, y_train, y_test = train_test_split(df2.iloc[:, :-1], 
                                                    df2.iloc[:, -1], 
                                                    test_size = 0.3, 
                                                    random_state = 42)

# %% [markdown]
# ###The first column is the index for each data point and hence we can simply remove it.

# %%
X_train = X_train.iloc[:, 1:]
X_test = X_test.iloc[:, 1:]

# %% [markdown]
# ### As we have cars with names but not with company names let's create new column for company i.e is brand

# %%
new_col_train = X_train["Name"].str.split(" ", expand = True)
new_col_test = X_test["Name"].str.split(" ", expand = True)

# %%
X_train["Brand"]=new_col_train[0]
X_test["Brand"]=new_col_test[0]

# %%
plt.figure(figsize = (12, 8))
plot = sns.countplot(x = 'Brand', data = X_train)
plt.xticks(rotation = 90)
for p in plot.patches:
    plot.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                        ha = 'center', 
                        va = 'center', 
                        xytext = (0, 5),
                        textcoords = 'offset points')

plt.title("Count of cars based on manufacturers")
plt.xlabel("Brand")
plt.ylabel("Count of cars")

# %%
X_train.dtypes

# %% [markdown]
# ### Notice that Mileage, Engine,Power ,Prices are not int or float, lets clean data and convert them

# %% [markdown]
# #### Mileage

# %%
mileage_train = X_train["Mileage"].str.split(" ", expand = True)
mileage_test = X_test["Mileage"].str.split(" ", expand = True)

X_train["Mileage"] = pd.to_numeric(mileage_train[0], errors = 'coerce')
X_test["Mileage"] = pd.to_numeric(mileage_test[0], errors = 'coerce')

# %% [markdown]
# ### Engine

# %%
cc_train = X_train["Engine"].str.split(" ", expand = True)
cc_test = X_test["Engine"].str.split(" ", expand = True)

X_train["Engine"] = pd.to_numeric(cc_train[0], errors = 'coerce')
X_test["Engine"] = pd.to_numeric(cc_test[0], errors = 'coerce')

# %% [markdown]
# #### Power

# %%
bhp_train = X_train["Power"].str.split(" ", expand = True)
bhp_test = X_test["Power"].str.split(" ", expand = True)

X_train["Power"] = pd.to_numeric(bhp_train[0], errors = 'coerce')
X_test["Power"] = pd.to_numeric(bhp_test[0], errors = 'coerce')

# %% [markdown]
# ####New_price
# 

# %%
X_train

# %%
X_train.dtypes

# %% [markdown]
# #### Let's Check is there any relation between some of the numerical columns

# %%
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style="darkgrid")
sns.scatterplot(data=X_train, x="Mileage", y="New_Price")

# %%
sns.scatterplot(data=X_train, x="Mileage", y="Engine")


# %%
sns.scatterplot(data=X_train, x="Mileage", y="Power")


# %%
sns.scatterplot(data=X_train, x="Engine", y="Power")


# %% [markdown]
# ### Listing all Brands of cars

# %%
s=list(X_train['Brand'].unique())
s.sort()
print(len(s),s)

# %% [markdown]
# ### Checking for Null Values

# %%
X_train.isna().sum()

# %% [markdown]
# ### Filling the missing values with mean of their respective column As they are numerical

# %%
X_train["Engine"].fillna(X_train["Engine"].astype("float64").mean(), inplace = True)
X_test["Engine"].fillna(X_train["Engine"].astype("float64").mean(), inplace = True)

X_train["Mileage"].fillna(X_train["Mileage"].astype("float64").mean(), inplace = True)
X_test["Mileage"].fillna(X_train["Mileage"].astype("float64").mean(), inplace = True)

X_train["Power"].fillna(X_train["Power"].astype("float64").mean(), inplace = True)
X_test["Power"].fillna(X_train["Power"].astype("float64").mean(), inplace = True)

X_train["Seats"].fillna(X_train["Seats"].astype("float64").mean(), inplace = True)
X_test["Seats"].fillna(X_train["Seats"].astype("float64").mean(), inplace = True)


# %%
X_train.isna().sum()

# %% [markdown]
# #### lets drop New Price column because it doesnot have impact on the 

# %%
X_train.drop(["New_Price"], axis = 1, inplace = True)
X_test.drop(["New_Price"], axis = 1, inplace = True)

# %% [markdown]
# #### Now that we have worked with the training data, let's create dummy columns for categorical columns before we begin training.

# %%
X_train = pd.get_dummies(X_train,columns = ["Brand", "Fuel_Type", "Transmission", "Owner_Type"],
                         drop_first = True)
X_test = pd.get_dummies(X_test,columns = ["Brand", "Fuel_Type", "Transmission", "Owner_Type"],
                         drop_first = True)

# %%
X_train.isna().sum()

# %% [markdown]
# #### If some of the categorical columns doesnot created in test dataset because of its less count of unique values,lets fill them

# %%
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]

# %%
X_train_df=X_train.copy()
X_test_df=X_test.copy()

# %%
X_train.drop(["Name"], axis = 1, inplace = True)
X_test.drop(["Name"], axis = 1, inplace = True)

# %%
X_train.drop(["Location"], axis = 1, inplace = True)
X_test.drop(["Location"], axis = 1, inplace = True)

# %% [markdown]
# #### Scaling the data

# %%
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)

# %% [markdown]
# #### Training and testing with  Linear Regression Model
# 
# *   List item
# *   List item
# 
# 

# %%
linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)
y_pred = linearRegression.predict(X_test)
r2_score(y_test, y_pred)

# %% [markdown]
# #### Training and testing with  Random Forest Model

# %%
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
r2_score(y_test, y_pred)

# %% [markdown]
# ### Let's Manually check these predictions

# %%
# Finally, let's manually check these predictions
# To obtain the actual prices, we take the exponential of the log_price
df_ev = pd.DataFrame((y_pred), columns=['Predicted Price'])

# We can also include the Actual price column in that data frame (so we can manually compare them)
y_test = y_test.reset_index(drop=True)
df_ev['Actual Price'] = (y_test)

# we can calculate the difference between the targets and the predictions
df_ev['Residual'] = df_ev['Actual Price'] - df_ev['Predicted Price']
df_ev['Difference%'] = np.absolute(df_ev['Residual']/df_ev['Actual Price']*100)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_ev.sort_values(by=['Difference%'])

df_ev

# %%
X_train_df.describe(include='all')

# %%



