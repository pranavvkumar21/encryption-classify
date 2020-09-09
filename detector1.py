import os
import csv
import tensorflow as tf
import numpy as np
import keras
from keras.models import model_from_json

np.random.seed(0)                                   #for getting consistent results
ampmax =44                                          #for normalizing the dataset

#randomly shuffle dataset and labels for better generalization using mini-batches
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b

#get the names of all files in the given folder
def getlist(filepath):
    list= os.listdir(filepath)
    list.sort()
    return list

#convert list to float
def convert_to_float(eg):
    for i in range(len(eg)):
        for j in range(len(eg[i])):
            eg[i][j]=np.float(eg[i][j].replace(",","."))
    return eg

#get data from files and normalize
def optdata(folder, ampmax=44):
    totallen = len(getlist(folder[0]))+len(getlist(folder[1]))+len(getlist(folder[2]))
    count=0
    train_label = []
    X=np.empty([totallen,400])
    for i in range(0,len(folder)):
        file_list = getlist(folder[i])
        for filename in file_list:
            File = open(folder[i]+"/"+filename)
            filedata = list(csv.reader(File, delimiter=";"))
            example = np.array(convert_to_float(filedata[15:]))
            train_label.append(i)
            X[count]=example[:,1]
            count=count+1
    X[:,0] =X[:,0]/ ampmax
    X,train_label=shuffle_in_unison(X, train_label)
    print(X.shape)
    return X,train_label
initializer = tf.keras.initializers.GlorotNormal()
#keras sequential model for multilayer-perceptron
model = keras.Sequential([
    keras.layers.core.Dense(350,input_shape=(400,), activation='relu'),
    #keras.layers.core.Dense(240, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(220, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(180, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(175, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(160, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(155, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(150, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(145, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(140, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(135, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(130, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(125, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(120, activation='relu'),
    keras.layers.core.Dropout(0.2),


    keras.layers.core.Dense(120, activation='relu'),

    keras.layers.core.Dense(30, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(20, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(10, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(7, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(5, activation='relu'),
    keras.layers.core.Dropout(0.2),
    keras.layers.core.Dense(3,activation='softmax')
    #keras.layers.core.Dropout(0.4),
    #keras.layers.core.Softmax()
])

#train the dataset
def train_data():
    #folder_train = input("\n\n\n\n\n\n\nenter train folder names of training set seperated by semicolons  -->")
    folder_train=["3DES","AES128","AES256"]
    #folder_train= folder_train.split(";")
    X_train,y_train=optdata(folder_train,ampmax)
    opt = keras.optimizers.Adam(learning_rate=0.000003)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.fit(X_train,y_train, epochs=60, batch_size=1)

#input path
train_data()
#sfolder_test = input("\nenter train folder names of test set seperated by semicolons  -->")
folder_test=["convert_3DES","convert_AES128","convert_AES256"]

X_test, y_test = optdata(folder_test,ampmax)
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print("accuracy = "+str(test_acc))

#probability_model = tf.keras.Sequential([model,
#                                         tf.keras.layers.core.Softmax()])
test_images,ylabel = optdata(folder_test)
predictions = model.predict(test_images)
lab = []
for i in range(0,len(predictions)):
    lab.append(np.argmax(predictions[i]))
print(ylabel)
print(lab)
