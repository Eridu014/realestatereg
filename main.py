import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

######################################
#retrive the data set
######################################

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
housing = load_housing_data()
df = pd.DataFrame(housing)
#print(df['ocean_proximity'].value_counts())
#(df.describe())

############################################
#Create a Test Set
############################################

#set aside test set
def split_train_set(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.inloc[test_indices]
    
#compute hash of instance's identifier to ensure stable train/test split
#def test_set_check(identifier, test_ratio):
    #return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

#def split_train_set_by_id(data, test_ratio, id_column):
    #ids = data[id_column]
    #in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    #return data.iloc[~in_test_set], data.iloc[in_test_set]
#housing data does not have indentifier(primary key) for each tuple of the dataset

#We could use the row index as the ID for each instance
#housing_with_id = housing.reset_index()
#above code adds an index column to the housing dataset
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
    
#You could create an idenitifier utilizing stable features of the dataset. Ex) Latitide and Longitude
#housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id") 
    
#utilizing SciKit
train_test, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

#Stratafied Sampling
#create strata of median incomes
housing["income_cat"] = pd.cut(housing["median_income"],
                              bins = [0., 1.5, 3.0, 4.5, 6., np.inf], #income levels (e.g. 1.5 = 15k)
                              labels = [1, 2, 3, 4, 5]) #labels for incomes
#show income categories histogram
#df["income_cat"].hist()

#take stratified sample of income
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
    
#income category proportions
#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

#remove the income_Cat attribute so data can revert back to original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)
    
    
###################################
#Discovering and Visualizing Data to Gain Insights
###################################
#create copy as to not harm the original training set
housing = strat_train_set.copy()

#scattter plot for geo location
#housing.plot(kind = "scatter", x = "longitude", y = "latitude")
#set alpha parameter to 0.1 to view density of data points
#housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)

#housing prices and pop scatterplot
#housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.5, s = housing["population"]/100, label = "population", figsize = (11, 7), c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True)
#plt.legend()

#compute the standard correlation coefficient

#print(corr_matrix["median_house_value"].sort_values(ascending = True))
corr_matrix = housing.corr()
#create histogram
#df.hist(bins = 50, figsize = (20, 15))
#plt.show()

#using Pandas, create scatter matrix of all numeric attributes correlated with every other numeric attribute
attributes = ["median_house_value", "total_rooms", "median_income", "housing_median_age"]
scatter_matrix(housing[attributes], figsize = (12, 8))
plt.show()

#correlation between median income and median home price
housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)
plt.show()

#creating new attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
print(corr_matrix["rooms_per_household"].sort_values(ascending = False))



###################################################
#PREPARING DATA FOR ML ALGORITHMS
###################################################

housing2 = strat_train_set.drop("median_house_value", axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()

######################################################
#DATA CLEANING
######################################################

#get rid of missing features in the dataset
#housing2.dropna(subset = ["total_bedrooms"]) #option 1
#housing2.drop("total_bedrooms", axis = 1) #option 2
#median = housing2["total_bedrooms"].median() #option 3
#housing2["total_bedrooms"].fillna(median, inplace = True)

#or Scikit
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")

housing2_num = housing2.drop("ocean_proximity", axis = 1) #drop non-numerical attribute 
imputer.fit(housing2_num)
print(imputer.statistics_) #median of each attribute stored in statistics_ variable

#replace missing values by learned medians
X = imputer.transform(housing2_num)

#reconvert back into pandas DF (rn its just a numpy array)
housing_tr = pd.DataFrame(X, columns = housing2_num.columns, index = housing2_num.index)

#remake ocean_proximity attribute (categorical) into a numerical attribute
housing_cat = housing2[["ocean_proximity"]]
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
#one-hot encoding (assigning binary attribute per category e.g. NEAR OCEAN = 1)
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#the variable is a sparse matrix instead of a numpy array

#CUSTOM TRANSFORMER
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room #add_bedrooms_per_room is hyperparameter of transformer
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]