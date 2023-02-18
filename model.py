import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pickle

data = pd.read_csv('housing.csv')
# select only 1000 rows
data = data[:1000]
# converting categorical column to int datatype
label_encoder = LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

# filling in missing values
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

# making data a numpy array like
x = data.drop(['median_house_value'], axis=1)
y = data.median_house_value
x = x.values
y = y.values
# dividing data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

# standardzing the data
stds = StandardScaler()
scaler = stds.fit(x_train)
rescaledx = scaler.transform(x_train)

# selecting and fitting the model for training
model = RandomForestRegressor()
model.fit(rescaledx, y_train)
# saving the trained mode
pickle.dump(model, open('rf_model.pkl', 'wb'))
# saving StandardScaler
pickle.dump(stds, open('scaler.pkl', 'wb'))
