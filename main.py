import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
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
df["income_cat"].hist()

#take stratified sample of income
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
    
#income category proportions
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

#remove the income_Cat attribute so data can revert back to original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)
    
    
###################################
#Discovering and Visualizing Data to Gain Insights
###################################
#create copy as to not harm the original training set
housing = strat_train_set.copy()

#scattter plot for geo location
housing.plot(kind = "scatter", x = "longitude", y = "latitude")
#set alpha parameter to 0.1 to view density of data points
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)

#housing prices and pop scatterplot
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.5, s = housing["population"]/100, label = "population", figsize = (11, 7), c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True)
plt.legend()

#compute the standard correlation coefficient
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = True)
#create histogram
df.hist(bins = 50, figsize = (20, 15))
plt.show()