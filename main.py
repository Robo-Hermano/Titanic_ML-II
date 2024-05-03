#NOTE: like my other titanic code, I advise using jupyter / google colab, and running them the way I've blocked them

#block 1
#import modules
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#get in our datasets
trainset = pd.read_csv("space_titanic_train.csv")
trainset = trainset.reindex(np.random.permutation(trainset.index))
testset = pd.read_csv("space_titanic_test.csv")
testids = testset["PassengerId"]
trainset = trainset.bfill()
testset = testset.bfill()

#block 2
#get descriptions of the trainset
print(trainset.columns)
print(trainset.dtypes)
print(trainset.head())
#more descriptions
print(trainset["HomePlanet"].unique())
print(trainset["Destination"].unique())
print(trainset.shape)
print(trainset["Cabin"].iloc[:5])

#block 3
#engineering the cabin column to be more useful
trainset["Deck"] = trainset["Cabin"]
trainset["Deck_Side"] = trainset["Cabin"]
testset["Deck"] = testset["Cabin"]
testset["Deck_Side"] = testset["Cabin"]
trainset["Deck_Num"] = trainset["Cabin"]
testset["Deck_Num"] = testset["Cabin"]
def deck_func(value):
    return value[0]
def deck_side_func(value):
    return value[-1]
def deck_num_func(value):
    arr = [i for i in value if i in ["1","2","3","4","5","6","7","8","0","9"]]
    return int(''.join(arr))
trainset["Deck_Num"] = trainset["Deck_Num"].apply(deck_num_func)
testset["Deck_Num"] = testset["Deck_Num"].apply(deck_num_func)
trainset["Deck"] = trainset["Deck"].apply(deck_func)
testset["Deck"] = testset["Deck"].apply(deck_func)
trainset["Deck_Side"] = trainset["Deck_Side"].apply(deck_side_func)
testset["Deck_Side"] = testset["Deck_Side"].apply(deck_side_func)
print(trainset.head())

#block 4
#encode categorical values
dummycols_train = pd.get_dummies(trainset[["Deck","HomePlanet","Destination"]])
dummycols_test = pd.get_dummies(testset[["Deck","HomePlanet","Destination"]])
trainset = trainset.drop(columns=["Deck","HomePlanet","Destination"])
testset = testset.drop(columns=["Deck","HomePlanet","Destination"])
trainset = pd.concat([trainset, dummycols_train])
testset = pd.concat([testset, dummycols_test])
#get rid of nan values
trainset = trainset.fillna(method="pad")
trainset = trainset.fillna(False)
testset = testset.fillna(method="pad")
testset = testset.fillna(False)
#transform to int so more useable for correlationd
def bool_transform(value):
    if value == False or value == "S":
        return 0
    elif value == True or value == "P":
        return 1
    else:
        return 0
for i in trainset.columns:
    if trainset[i].dtype == "bool" or i == "Deck_Side":
        trainset[i] = trainset[i].apply(bool_transform)
for i in testset.columns:
    if testset[i].dtype == "bool" or i == "Deck_Side":
        testset[i] = testset[i].apply(bool_transform)
#drop unneeded columns
trainset = trainset.drop(columns=["Name","Cabin","PassengerId"])
testset = testset.drop(columns=["Name","Cabin","PassengerId"])
correlations = trainset.corr()
print(correlations["Transported"])

#block 5
#drop uncorrelating columns
trainset = trainset.drop(columns = ["CryoSleep", "VIP", "ShoppingMall", "Deck_Num", "Deck_A", "Deck_B", "Deck_C", "Deck_D", "Deck_E", "Deck_T", "HomePlanet_Mars", "Destination_55 Cancri e", "Destination_PSO J318.5-22"])
testset = testset.drop(columns = ["CryoSleep", "VIP", "ShoppingMall", "Deck_Num", "Deck_A", "Deck_B", "Deck_C", "Deck_D", "Deck_E", "Deck_T", "HomePlanet_Mars", "Destination_55 Cancri e", "Destination_PSO J318.5-22"])
#split into train and test
print(trainset[trainset.index.duplicated()])
trainset = trainset[~trainset.index.duplicated()]
trainset = trainset.reindex(np.random.permutation(trainset.index))
minitrain = trainset.iloc[:6000]
minivalidate = trainset.iloc[6000:]
#normalised input layer for model
input = {
    "Age": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "FoodCourt": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "Deck_Side": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "HomePlanet_Earth": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "RoomService": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "Destination_TRAPPIST-1e": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "HomePlanet_Europa": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "Spa": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "VRDeck": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "Deck_G": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    "Deck_F": tf.keras.layers.Input(shape=(1,), dtype=tf.float32),
    }
age = tf.keras.layers.Normalization(axis = None)
age.adapt(minitrain["Age"])
age = age(input.get("Age"))
foodcourt = tf.keras.layers.Normalization(axis = None)
foodcourt.adapt(minitrain["FoodCourt"])
foodcourt = foodcourt(input.get("FoodCourt"))
deckside = tf.keras.layers.Normalization(axis = None)
deckside.adapt(minitrain["Deck_Side"])
deckside = deckside(input.get("Deck_Side"))
earth = tf.keras.layers.Normalization(axis = None)
earth.adapt(minitrain["HomePlanet_Earth"])
earth = earth(input.get("HomePlanet_Earth"))
roomservice = tf.keras.layers.Normalization(axis = None)
roomservice.adapt(minitrain["RoomService"])
roomservice = roomservice(input.get("RoomService"))
trappist = tf.keras.layers.Normalization(axis = None)
trappist.adapt(minitrain["Destination_TRAPPIST-1e"])
trappist = trappist(input.get("Destination_TRAPPIST-1e"))
europa = tf.keras.layers.Normalization(axis = None)
europa.adapt(minitrain["HomePlanet_Europa"])
europa = europa(input.get("HomePlanet_Europa"))
spa = tf.keras.layers.Normalization(axis = None)
spa.adapt(minitrain["Spa"])
spa = spa(input.get("Spa"))
vrdeck = tf.keras.layers.Normalization(axis = None)
vrdeck.adapt(minitrain["VRDeck"])
vrdeck = vrdeck(input.get("VRDeck"))
deckg = tf.keras.layers.Normalization(axis = None)
deckg.adapt(minitrain["Deck_G"])
deckg = deckg(input.get("Deck_G"))
deckf = tf.keras.layers.Normalization(axis = None)
deckf.adapt(minitrain["Deck_F"])
deckf = deckf(input.get("Deck_F"))

#block 6
#creating and compiling deep neural network model
input_layer = tf.keras.layers.Concatenate()([age, foodcourt, deckside, earth, roomservice, trappist, europa, spa, vrdeck, deckg, deckf])
dense_output = tf.keras.layers.Dense(units=256, activation = 'relu', name = 'hidden_dense_layer_1')(input_layer)
dense_output = tf.keras.layers.Dense(units=128, activation = 'relu', name = 'hidden_Dense_layer_2')(dense_output)
dense_output = tf.keras.layers.Dense(units=64, activation = 'relu', name = 'hidden_dense_layer_3')(dense_output)
dense_output = tf.keras.layers.Dense(units=32, activation = 'relu', name = 'hidden_Dense_layer_4')(dense_output)
dense_output = tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'dense_output')(dense_output)
outputs = {"dense_output": dense_output}
mini_modella = tf.keras.Model(inputs = input, outputs = outputs)
mini_modella.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1),
    loss = "binary_crossentropy",
    metrics = ["accuracy"]  
)

#block 7
print(minitrain.isnull().sum())
#setup x and y as usual
xtrain = minitrain.drop(columns="Transported")
xtest = minivalidate.drop(columns=["Transported"])
trainfeatures = {name:np.array(value) for name, value in minitrain.items()}
testfeatures = {name:np.array(value) for name, value in minivalidate.items()}
history = mini_modella.fit(x=trainfeatures,y=minitrain["Transported"], batch_size = 3000, epochs = 300, validation_split = 0.25)
#use kfold cross validation here, might generate more reliable test accuracy
#using matplotlib this time to plot and visualise results
epochs = history.epoch
hist = pd.DataFrame(history.history)
mse = hist["accuracy"]
plt.figure()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(epochs, mse, label="Training loss")
plt.plot(epochs, history.history["val_accuracy"], label = "Test loss")
mergedmse = mse.tolist() + history.history["val_accuracy"]
plt.ylim([min(mergedmse)*0.97, max(mergedmse)*1.03])
plt.legend()
plt.show()
mini_modella.evaluate(x = testfeatures, y = minivalidate["Transported"], return_dict = True)

#block 8
#use model to predict for testset
testingfeatures = {name:np.array(value) for name, value in testset.items()}
for i in testingfeatures:
    testingfeatures[i] = testingfeatures[i][0:4277]
predictions = mini_modella.predict(testingfeatures)
print(predictions)

#block 9
#convert to boolean column format
predictions = predictions["dense_output"]
print(min(predictions))
preds_bool = np.round(predictions).astype(bool)
test_transported = []
for i in preds_bool:
    test_transported.append(i[0])
print(test_transported)

#block 10
#output with testset ids to submission model
print(test_transported)
print(len(test_transported), len(testids))
final_output = pd.DataFrame({'PassengerId': testids, 'Transported': test_transported})
final_output.to_csv('model_preds.csv', index=False)
