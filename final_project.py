import pandas as pd
import numpy as np

# 1. read data

ltable = pd.read_csv("ltable.csv")
rtable = pd.read_csv("rtable.csv")
train = pd.read_csv("train.csv")


# 2. blocking
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR


def block_by_brand(ltable, rtable):
    # ensure brand is str
    ltable['brand'] = ltable['brand'].astype(str)
    rtable['brand'] = rtable['brand'].astype(str)

    # get all brands
    brands_l = set(ltable["brand"].values)
    brands_r = set(rtable["brand"].values)
    brands = brands_l.union(brands_r)

    # map each brand to left ids and right ids
    brand2ids_l = {b.lower(): [] for b in brands}
    brand2ids_r = {b.lower(): [] for b in brands}  #{'a': [], 'b': [], 'c': []}
    for i, x in ltable.iterrows():
        brand2ids_l[x["brand"].lower()].append(x["id"])
    for i, x in rtable.iterrows():
        brand2ids_r[x["brand"].lower()].append(x["id"])

    # put id pairs that share the same brand in candidate set
    candset = []
    for brd in brands:
        l_ids = brand2ids_l[brd]
        r_ids = brand2ids_r[brd]
        for i in l_ids:
            for j in r_ids:
                #if ltable['category'][i] == rtable['category'][j]:   
                candset.append([i, j])
    return candset
# blocking to reduce the number of pairs to be compared
candset = block_by_brand(ltable, rtable)
#print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
#print("number of pairs after blocking",len(candset))
candset_df = pairs2LR(ltable, rtable, candset)
candset_df.to_csv("filter.csv", index=False)

# candset_df =pd.DataFrame(candset_df)
# for index, row in candset_df.iterrows():
#     if(row['category_l'] != row['category_r']):
#         candset_df = candset_df.drop(index, axis=0)

# 3. Feature engineering
import Levenshtein as lev

#change
def price_difference_high(row):
    x = row["price" + "_l"].lower()
    y = row["price" + "_r"].lower()
    x = float(x)
    y = float(y)
    return (max(x,y) - min(x,y)) / (max(x,y) + 0.1)

def price_difference_low(row):
    x = row["price" + "_l"].lower()
    y = row["price" + "_r"].lower()
    x = float(x)
    y = float(y)
    return (max(x,y) - min(x,y)) / (min(x,y) + 0.1)

def whether_same_category(row):
    x = row["category" + "_l"].lower()
    y = row["category" + "_r"].lower()
    return(x == y)

def whether_same_title(row):
    x = row["title" + "_l"].lower()
    y = row["title" + "_r"].lower()
    return(x == y)

def similarity(row, attr):
    if(attr != "price"):
        x = set(row[attr + "_l"].lower().split())
        y = set(row[attr + "_r"].lower().split())
        result = len(x.intersection(y)) / max(len(x), len(y))
    if(attr == "price"):
        result = int(price_difference_high(row))
    return result
 #jaccard_similarity for "title", "brand", "modelno",  price_difference_high for "price", whether_same_category for "category" 

def distance(row, attr):
    if(attr != "price"):
        x = row[attr + "_l"].lower()
        y = row[attr + "_r"].lower()
        result = lev.distance(x, y)
    if(attr == "price"):
        result = (price_difference_low(row))
    return result
### levenshtein_distance

from sklearn import preprocessing

def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "brand", "modelno", "price"]  #change
    features = []
    for attr in attrs:
        j_sim = LR.apply(similarity, attr=attr, axis=1)
        l_dist = LR.apply(distance, attr=attr, axis=1)
        features.append(j_sim)
        features.append(l_dist)        
    features = np.array(features).T
    features = preprocessing.normalize(features)   # change: normalization
    return features
candset_features = feature_engineering(candset_df)

# also perform feature engineering to the training set
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = feature_engineering(training_df)
training_label = train.label.values

# 4. Model training and prediction
from sklearn.ensemble  import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

model = GaussianNB()
cross_val = KFold(n_splits=5)
scores = cross_val_score(model, training_features, training_label, cv=cross_val, scoring='roc_auc')
print(scores)
print(scores.mean())

#"Training and Testing Machine Learning Models" by Alex Strebeck

model = GaussianNB()
model.fit(training_features, training_label)
print(model.score(training_features, training_label))
y_pred = model.predict(candset_features)

# 5. output

matching_pairs = candset_df.loc[y_pred == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
Ref = [pair for pair in matching_pairs if
              pair in matching_pairs_in_training]  # remove the matching pairs already in training
print(len(Ref))
print(len(pred_pairs))
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)