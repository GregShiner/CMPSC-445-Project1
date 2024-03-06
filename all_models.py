# %% [markdown]
# First, install dependencies and download the data

# %%
#%pip install -r requirements.txt
#!curl -l -o data.csv "https://phl.carto.com/api/v2/sql?q=SELECT+*,+ST_Y(the_geom)+AS+lat,+ST_X(the_geom)+AS+lng+FROM+opa_properties_public&filename=opa_properties_public&format=csv&skipfields=cartodb_id"

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

# %%
df = pd.read_csv('data.csv')

# %%
df.shape

# %%
df.head()

# %%
df.columns

# %% [markdown]
# Here we can see all of the columns available in the dataset. Many of these columns are not really necesary. For example, geographical data such as street addresses are not very useful because they don't contain any categorical or numerical data that can be easily consumed by a linear regression mode.

# %%
COLUMNS_TO_REMOVE = """
the_geom
the_geom_webmercator
beginning_point
book_and_page
building_code_description
category_code_description
cross_reference
geographic_ward
house_number
location
mailing_address_1
mailing_address_2
mailing_care_of
mailing_city_state
mailing_street
mailing_zip
other_building
owner_1
owner_2
parcel_number
registry_number
state_code
street_code
street_designation
street_direction
street_name
suffix
assessment_date
recording_date
sale_date
pin
zip_code
parcel_shape
quality_grade
building_code
building_code_new
building_code_description_new
objectid
unit
"""
df_first_column_drop = df.drop(columns=COLUMNS_TO_REMOVE.split())

# %%
df_first_column_drop.to_csv('data_first_clean.csv', index=False)

# %% [markdown]
# One thing to note about this dataset is that this data is for all types of properties in Philadelphia. Since we are only interested in housing data, we will filter this data to only be for housing type properties. This is accomplished by filtering for properties with a category code of 1 (single family), 2 (multi family), or 3 (mixed).

# %%
# Get all unique values of (category_code, category_code_description)
df[['category_code', 'category_code_description']].drop_duplicates().sort_values(by='category_code')

# %%
# Print number of data points
# print(df_first_column_drop.shape)
# Filter category-type to 1, or 2
df_first_column_drop = df_first_column_drop[df_first_column_drop['category_code'] <= 2]
# Print number of data points
# print(df_first_column_drop.shape)

# %% [markdown]
# Now that we have removed some of the columns, lets see what we are left with.

# %%
df_first_column_drop.dtypes

# %% [markdown]
# We still have a lot of columns. It is very likely that many more of these will get dropped for the following reasons:
# 1. Too many missing values
# 2. Too many unique values
# 3. Not enough correlation with the target variable
# 
# Let's start by dealing with the first case: too many missing values. We will check this by counting the number of missing values in each column and sorting by this number.
# 
# There are also a few columns that seem to have a default value put in them instead of being left blank. Lets also change these default values to na, so they can be counted correctly.

# %%
df_first_column_drop['depth'] = df_first_column_drop['depth'].replace(0, np.nan)
df_first_column_drop['total_area'] = df_first_column_drop['total_area'].replace(0, np.nan)
df_first_column_drop['total_livable_area'] = df_first_column_drop['total_livable_area'].replace(0, np.nan)
df_first_column_drop['year_built'] = df_first_column_drop['year_built'].replace(0, np.nan)
df_first_column_drop['sale_price'] = df_first_column_drop['sale_price'].replace(1, np.nan)

# %%
"""
We still have a lot of columns. It is very likely that many more of these will get dropped for the following reasons:
1. Too many missing values
2. Too many unique values
3. Not enough correlation with the target variable

Let's start by dealing withe the first case: too many missing values. We will check this by counting the number of missing values in each column and sorting by this number.
"""
# print(df_first_column_drop.isna().sum().sort_values(ascending=False))
# Print the number of missing values and the percentage of missing values
missing_values = df_first_column_drop.isna().sum().sort_values(ascending=False)
missing_values = missing_values[missing_values > 0]
missing_values = pd.DataFrame(missing_values, columns=['missing_values'])
missing_values['percentage_missing'] = missing_values['missing_values'] / len(df_first_column_drop)
# print(missing_values)

# %% [markdown]
# Now some of these pieces of missing data won't be that big of a deal because we can either fill or impute the data. We can also drop rows with missing data, but we should do this sparingly. For some categorical columns, you could set a default value, but I will not be doing this much. You cannot assume exactly what the person who entered the data intended by leaving it blank, and filling it could cause innacuracies, especially with columns with a lot of missing data. But for some columns, there is simply too much missing data. For this reason, I will be dropping all columns with more than 25% missing data.

# %%
COLUMNS_TO_REMOVE_MISSING_VALUES = missing_values[missing_values['percentage_missing'] > 0.25].index
df_second_column_drop = df_first_column_drop.drop(columns=COLUMNS_TO_REMOVE_MISSING_VALUES)

# %%
df_second_column_drop.to_csv('data_second_clean.csv', index=False)

# %% [markdown]
# Now let's recheck the list of columns with missing data, and address each one indiviually.

# %%
missing_values_2 = df_second_column_drop.isna().sum().sort_values(ascending=False)
missing_values_2 = missing_values_2[missing_values_2 > 0]
missing_values_2 = pd.DataFrame(missing_values_2, columns=['missing_values'])
missing_values_2['percentage_missing'] = missing_values_2['missing_values'] / len(df_second_column_drop)
# print(missing_values_2)

# %% [markdown]
# Now that we are left with columns with most of their data filled in, it is much safter to start filling in with default or imputed data.

# %%
from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(strategy='median')
mode_imputer = SimpleImputer(strategy='most_frequent')
df_fill = df_second_column_drop
# year_built_estimate can be filled with N's
df_fill['year_built_estimate'] = df_fill['year_built_estimate'].fillna('N')
# garage_spaces can just be filled with 0. It is a fair assumption that if the value is missing, there is no garage
df_fill['garage_spaces'] = df_fill['garage_spaces'].fillna(0)
# fireplaces can just be filled with 0. It is a fair assumption that if the value is missing, there are no fireplaces
df_fill['fireplaces'] = df_fill['fireplaces'].fillna(0)
# number of bathrooms can be filled with the median
df_fill['number_of_bathrooms'] = median_imputer.fit_transform(df_fill[['number_of_bathrooms']])
# interior condition can be filled with the median
df_fill['interior_condition'] = median_imputer.fit_transform(df_fill[['interior_condition']])
# exterior condition can be filled with the median
df_fill['exterior_condition'] = median_imputer.fit_transform(df_fill[['exterior_condition']])
# number of bedrooms can be filled with the median
df_fill['number_of_bedrooms'] = median_imputer.fit_transform(df_fill[['number_of_bedrooms']])
# number of stories can be filled with the median
df_fill['number_stories'] = median_imputer.fit_transform(df_fill[['number_stories']])
# NOTE: Maybe change these to mode
# general construction can be filled with a new category called 'unknown'
df_fill['general_construction'] = df_fill['general_construction'].fillna('unknown')
# quality grade will be skipped for now because it needs to be transformed into a numeric column
# year built can be filled with the median
df_fill['year_built'] = median_imputer.fit_transform(df_fill[['year_built']])
# total livable area can be filled with the median
df_fill['total_livable_area'] = median_imputer.fit_transform(df_fill[['total_livable_area']])
# topography can be filled with a new category called 'unknown'
df_fill['topography'] = df_fill['topography'].fillna('unknown')
# depth can be filled with the median
df_fill['depth'] = median_imputer.fit_transform(df_fill[['depth']])
# total area can be filled with the median
df_fill['total_area'] = median_imputer.fit_transform(df_fill[['total_area']])
# view type can be filled with a new category called 'unknown'
df_fill['view_type'] = df_fill['view_type'].fillna('unknown')
# off street open can be filled with the median
df_fill['off_street_open'] = median_imputer.fit_transform(df_fill[['off_street_open']])
# frontage can be filled with the median
df_fill['frontage'] = median_imputer.fit_transform(df_fill[['frontage']])
# zoning can be filled with a new category called 'unknown'
df_fill['zoning'] = df_fill['zoning'].fillna('unknown')
# census tract can be filled with the median
df_fill['census_tract'] = median_imputer.fit_transform(df_fill[['census_tract']])
# lat and lng can be filled with the median
df_fill['lat'] = median_imputer.fit_transform(df_fill[['lat']])
df_fill['lng'] = median_imputer.fit_transform(df_fill[['lng']])
# taxable building can be filled with the median
df_fill['taxable_building'] = median_imputer.fit_transform(df_fill[['taxable_building']])
# exempt land can be filled with the median
df_fill['exempt_land'] = median_imputer.fit_transform(df_fill[['exempt_land']])
# exempt building can be filled with the median
df_fill['exempt_building'] = median_imputer.fit_transform(df_fill[['exempt_building']])
# taxable land can be filled with the median
df_fill['taxable_land'] = median_imputer.fit_transform(df_fill[['taxable_land']])
# NOTE: Maybe change this to drop
# market value can be filled with the median
df_fill['market_value'] = median_imputer.fit_transform(df_fill[['market_value']])

# %% [markdown]
# Lets check again our missing data

# %%
missing_values_3 = df_fill.isna().sum().sort_values(ascending=False)
missing_values_3 = missing_values_3[missing_values_3 > 0]
missing_values_3 = pd.DataFrame(missing_values_3, columns=['missing_values'])
missing_values_3['percentage_missing'] = missing_values_3['missing_values'] / len(df_second_column_drop)
# print(missing_values_3)

# %%
# Save to csv
df_fill.to_csv('data_filled.csv', index=False)

# %% [markdown]
# Now all

# %%
# One hot columns
ONE_HOT_COLUMNS = [
    'category_code',
    'general_construction',
    'topography',
    'view_type',
    'zoning',
]

BINARY_COLUMNS = [
    'year_built_estimate',
    'homestead_exemption',
    'exempt_building',
    'exempt_land'
]

# %%
# One hot encode the columns
df_one_hot = pd.get_dummies(df_fill, columns=ONE_HOT_COLUMNS)
# Binary encode the columns
df_one_hot['year_built_estimate'] = df_one_hot['year_built_estimate'].map({'Y': True, 'N': False})
# Fillna with False
df_one_hot['year_built_estimate'] = df_one_hot['year_built_estimate'].fillna(False)
df_one_hot['homestead_exemption'] = df_one_hot['homestead_exemption'].map({80000: True, 0: False})
# Fillna with False
df_one_hot['homestead_exemption'] = df_one_hot['homestead_exemption'].fillna(False)
# Exempt building should be false if 0, else true
df_one_hot['exempt_building'] = df_one_hot['exempt_building'].map({0: False})
# Fillna with True
df_one_hot['exempt_building'] = df_one_hot['exempt_building'].fillna(True)
# Exempt land should be false if 0, else true
df_one_hot['exempt_land'] = df_one_hot['exempt_land'].map({0.0: False})
# Fillna with True
df_one_hot['exempt_land'] = df_one_hot['exempt_land'].fillna(True)

# %%
# Check that all columns are numeric
df_one_hot.dtypes.value_counts()

# %% [markdown]
# That is a lot of boolean columns. We can probably drop some of these columns, but we will do that later. For now, we will just convert these columns to 0 and 1.

# %%
# Get columns with type bool
bool_columns = df_one_hot.select_dtypes(include=bool).columns
# Change type of columns to int
df_one_hot[bool_columns] = df_one_hot[bool_columns].astype(int)

# %%
# Delete outliers
df_outliers = df_one_hot
# Delete census_tract outliers
#df_outliers = df_outliers[df_outliers['census_tract'] < 500]
# Delete depth outliers < 200 and > 32
df_outliers = df_outliers[df_outliers['depth'] < 144]
df_outliers = df_outliers[df_outliers['depth'] > 32]
# Fireplace < 6
df_outliers = df_outliers[df_outliers['fireplaces'] < 6]
# Frontage < 140
df_outliers = df_outliers[df_outliers['frontage'] < 50]
# Garage spaces < 5
df_outliers = df_outliers[df_outliers['garage_spaces'] < 5]
# Market value < 10_000_000
df_outliers = df_outliers[df_outliers['market_value'] < 2_000_000]
# Number of bathrooms < 6
df_outliers = df_outliers[df_outliers['number_of_bathrooms'] < 6]
# Number of bedrooms < 15
df_outliers = df_outliers[df_outliers['number_of_bedrooms'] < 6]
# Number of stories < 6
df_outliers = df_outliers[df_outliers['number_stories'] < 6]
# Taxable building < 4_000_000
df_outliers = df_outliers[df_outliers['taxable_building'] < 1_000_000]
# Taxable land < 1_000_000
df_outliers = df_outliers[df_outliers['taxable_land'] < 200_000]
# Total area < 250_000
df_outliers = df_outliers[df_outliers['total_area'] < 16_000]
# Total livable area < 250_000
df_outliers = df_outliers[df_outliers['total_livable_area'] < 8_000]
# Year built > 1840
df_outliers = df_outliers[df_outliers['year_built'] > 1890]

# %%
# Export to CSV excluding columns starting with ONE_HOT_COLUMNS
df_out = df_outliers
for column in ONE_HOT_COLUMNS:
    df_out = df_out.loc[:, ~df_out.columns.str.startswith(column)]
df_out.to_csv('data_outliers.csv', index=False)

#df_outliers.to_csv('data_outliers.csv', index=False)
df_out.shape

# %%
# Create a new column called "bed+bath" which is the sum of the number of bedrooms and bathrooms
df_engineered = df_outliers
df_engineered['bed+bath'] = df_engineered['number_of_bedrooms'] + df_engineered['number_of_bathrooms']
# %%
# Create a correlation graphic agains the market value
correlation = df_outliers.corr()
# Print the correlation with the target variable
correlation['market_value'].sort_values(ascending=False)

# %%
# Drop all columns with an absolute correlation less than 0.15
#mv_correlations = correlation["market_value"]


# %% [markdown]
# Now that preprocessing is complete, we can move on to feature engineering. For this step we will be applying PCA to the data. With PCA, we will be having it reduce the dimension so that we retain 95% of the variance.

# %%
from sklearn.preprocessing import StandardScaler
# Standardize the Data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_engineered), columns=df_engineered.columns)

# %%
X = df_scaled.drop(columns='market_value')
y = df_scaled['market_value']

# %%
from sklearn.decomposition import PCA
PCA_VARIANCE = 0.95
# Create a PCA instance
pca = PCA(PCA_VARIANCE)
pca.fit(X)
# Transform the data
X_pca = pca.transform(X)
#print(X.shape)
X_pca.shape

# %%
# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
from sklearn.model_selection import train_test_split
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

# %%
# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# %%
# Score the model
logging.info(f"model = {model.score(X_test, y_test)}")

# %%
pca_model = LinearRegression()
pca_model.fit(X_pca_train, y_pca_train)
logging.info(f"pca_model = {pca_model.score(X_pca_test, y_pca_test)}")

# %%
# Create a decision tree model
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
logging.info(f"tree_model = {tree_model.score(X_test, y_test)}")

# %%
# Create a random forest model
forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
logging.info(f"forest_model = {forest_model.score(X_test, y_test)}")

# %%
# Create a decision tree model with PCA
pca_tree_model = DecisionTreeRegressor()
pca_tree_model.fit(X_pca_train, y_pca_train)
logging.info(f"pca_tree_model = {pca_tree_model.score(X_pca_test, y_pca_test)}")

# %%
# Create a random forest model with PCA
pca_forest_model = RandomForestRegressor()
pca_forest_model.fit(X_pca_train, y_pca_train)
logging.info(f"pca_forest_model = {pca_forest_model.score(X_pca_test, y_pca_test)}")

# %%
# Create a gradient boosting model
gradient_model = GradientBoostingRegressor()
gradient_model.fit(X_train, y_train)
logging.info(f"gradient_model = {gradient_model.score(X_test, y_test)}")

# %%
# Create a gradient boosting model with PCA
pca_gradient_model = GradientBoostingRegressor()
pca_gradient_model.fit(X_pca_train, y_pca_train)
logging.info(f"pca_gradient_model = {pca_gradient_model.score(X_pca_test, y_pca_test)}")

# %%
# Create a model with xgboost
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
logging.info(f"xgb_model = {xgb_model.score(X_test, y_test)}")

# %%
# Create a model with xgboost with PCA
pca_xgb_model = xgb.XGBRegressor()
pca_xgb_model.fit(X_pca_train, y_pca_train)
logging.info(f"pca_xgb_model = {pca_xgb_model.score(X_pca_test, y_pca_test)}")

# %%
# Save all of the models
import pickle
pickle.dump(model, open('linear_model.pkl', 'wb'))
pickle.dump(tree_model, open('tree_model.pkl', 'wb'))
pickle.dump(forest_model, open('forest_model.pkl', 'wb'))
pickle.dump(gradient_model, open('gradient_model.pkl', 'wb'))
pickle.dump(xgb_model, open('xgb_model.pkl', 'wb'))

# %%
import pickle
pickle.dump(pca_model, open(f'pca_{PCA_VARIANCE}_linear_model.pkl', 'wb'))
pickle.dump(pca_tree_model, open(f'pca_{PCA_VARIANCE}_tree_model.pkl', 'wb'))
pickle.dump(pca_forest_model, open(f'pca_{PCA_VARIANCE}_forest_model.pkl', 'wb'))
pickle.dump(pca_gradient_model, open(f'pca_{PCA_VARIANCE}_gradient_model.pkl', 'wb'))
pickle.dump(pca_xgb_model, open(f'pca_{PCA_VARIANCE}_xgb_model.pkl', 'wb'))

# %%
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
for n in range(1, 11):
    variance = n / 10
    logging.info(f"{variance=}")
    pca = PCA(variance)
    pca.fit(X)
    X_pca = pca.transform(X)
    X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    pca_model = LinearRegression()
    pca_model.fit(X_pca_train, y_pca_train)
    logging.info(f"{pca_model.score(X_pca_test, y_pca_test)=}")
    pca_tree_model = DecisionTreeRegressor()
    pca_tree_model.fit(X_pca_train, y_pca_train)
    logging.info(f"{pca_tree_model.score(X_pca_test, y_pca_test)=}")
    pca_forest_model = RandomForestRegressor()
    pca_forest_model.fit(X_pca_train, y_pca_train)
    logging.info(f"{pca_forest_model.score(X_pca_test, y_pca_test)=}")
    pca_gradient_model = GradientBoostingRegressor()
    pca_gradient_model.fit(X_pca_train, y_pca_train)
    logging.info(f"{pca_gradient_model.score(X_pca_test, y_pca_test)=}")
    pca_xgb_model = xgb.XGBRegressor()
    pca_xgb_model.fit(X_pca_train, y_pca_train)
    logging.info(f"{pca_xgb_model.score(X_pca_test, y_pca_test)=}")

# %%



