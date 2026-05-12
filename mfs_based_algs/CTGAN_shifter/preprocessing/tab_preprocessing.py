
from copy import deepcopy
from sklearn import preprocessing as pp






def check_types(data):
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    data = data.loc[:,~data.columns.duplicated()].copy()
    for c in data.columns:
        if (len(data[c].unique()) >= 15) & (data[c].dtypes != "str") & (data[c].dtypes != "O"):
            data[c] = data[c].astype(float)
    return data

def get_cat_cols(data):
    cat_cols = []
    for c in data.columns:
        if (data[c].dtypes == "str") | (data[c].dtypes == "O") | (data[c].dtypes == "bool") | (data[c].nunique() < 30):
            cat_cols.append(c)
    #cat_cols = cat_cols + [data.columns.to_list()[-1]]
    return cat_cols


def preprocess_df(data):
    new_data = deepcopy(data)
    cat_cols = get_cat_cols(data)
    for c in cat_cols:
        encoder = pp.LabelEncoder()
        encoder.fit(new_data[c].values)
        new_data[c] = encoder.transform(new_data[c].values)
    return new_data


def get_x_y(data):
   
    X = data[data.columns[:-1]].values
    y = data[data.columns[-1]].values

    return X, y