import os
import csv
import tensorflow as tf
import numpy as np
import keras
from keras.models import model_from_json
import sys
from prettytable import PrettyTable

np.random.seed(0)                                   #for getting consistent results
ampmax =44                                          #for normalizing the dataset

#randomly shuffle dataset and labels for better generalization using mini-batches
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b


print("\n\n\nloading.............10%")


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


print("\nloading.............20%")


#get data from files and normalize
def optdata(folder, ampmax=44, shuffle=True, getfiles=False):
    totallen =0
    for i in folder:
        totallen =totallen+ len(getlist(i))
    count=0
    train_label = []
    files = []
    X=np.empty([totallen,401])
    for i in range(0,len(folder)):
        file_list = getlist(folder[i])
        for filename in file_list:
            File = open(folder[i]+"/"+filename)
            filedata = list(csv.reader(File, delimiter=";"))
            example = np.array(convert_to_float(filedata[14:]))
            train_label.append(i)
            X[count]=example[:,1]
            count=count+1
            files.append(filename)
    X[:,0] =X[:,0]/ ampmax
    if shuffle==True:
        X,train_label=shuffle_in_unison(X, train_label)
    #print(X.shape)
    if getfiles == False:
        return X,train_label
    else:
        return X,train_label,files


print("\nloading.............30%")


#keras sequential model for multilayer-perceptron
model = keras.Sequential([
    keras.layers.core.Dense(350,input_shape=(401,), activation='relu'),
    keras.layers.core.Dense(220, activation='relu'),
    keras.layers.core.Dense(180, activation='relu'),
    keras.layers.core.Dense(120, activation='relu'),
    keras.layers.core.Dense(30, activation='relu'),
    keras.layers.core.Dense(20, activation='relu'),
    keras.layers.core.Dense(10, activation='relu'),
    keras.layers.core.Dense(5, activation='relu'),
    keras.layers.core.Dense(3,activation='softmax')

])
print("\nloading.............40%")


#train the dataset
def train_data(epoch=200):
    try:
        folder_train = input("\n\n\n\n\n\n\nenter train folder names of training set seperated by semicolons  -->")
        #folder_train=["3DES","AES128","AES256"]
        folder_train= folder_train.split(";")
        X_train,y_train=optdata(folder_train,ampmax)
        opt = keras.optimizers.Adam(learning_rate=0.000003)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        model.fit(X_train,y_train, epochs=epoch, batch_size=3)

    except:
        print("unble to train data")
    try:
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

    except:
        print("error saving model")


print("\nloading.............60%")

#testing the data using test set
def test():
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("model.h5")
    except:
        print("trouble loading data \n make sure the files model.json and model.h5 exists")
    try:
        folder_test = input("\n\n\nenter train folder names of test set seperated by semicolons  -->")
        folder_test = folder_test.split(";")
        test_data,ylabel = optdata(folder_test)
        predictions = model.predict(test_data)
        lab = []
        for i in range(0,len(predictions)):
            lab.append(np.argmax(predictions[i]))
        conf_mat = np.zeros((len(folder_test),len(folder_test)))
        acc_count=0
        for i in range(0,len(lab)):
            if(ylabel[i]==lab[i]):
                acc_count=acc_count+1
            conf_mat[lab[i],ylabel[i]]= conf_mat[lab[i],ylabel[i]]+1
        precision =[]
        recall = []
        F1score = []
        for i in range(len(folder_test)):
            precision.append(conf_mat[i,i]/sum(conf_mat[i]))
            recall.append(conf_mat[i,i]/sum(conf_mat[:,i]))
            F1score.append((2*precision[i]*recall[i])/(precision[i]+recall[i]))

        print("\n\n accuracy = "+str(acc_count/len(lab)))

        confmat = PrettyTable()
        confmat.add_column("predicted \\ actual", folder_test)
        for i in range(len(folder_test)):
            confmat.add_column(folder_test[i],conf_mat[:,i])
        print("\n\n confusion matrix: \n\n")
        print(confmat)
        datatable = PrettyTable()
        tablecolnames= ["classes","precision","recall","F1 score"]
        datatable.add_column(tablecolnames[0],folder_test)
        datatable.add_column(tablecolnames[1],precision)
        datatable.add_column(tablecolnames[2],recall)
        datatable.add_column(tablecolnames[3],F1score)
        print("\n\ndata table: \n\n")
        print(datatable)


    except:
        print("error testing data. make sure the model was trained on the same number of encryptions")

print("\nloading.............80%")

#predict the new data given
def predict():
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
    except:
        print("error loading files \n make sure files exist")
    try:
        folder_predict = [input("\n\n\nenter train folder name of prediction set  -->")]
        test_data,_, files= optdata(folder_predict,shuffle=False,getfiles=True)
        predictions = model.predict(test_data)
        lab = []
        for i in range(0,len(predictions)):
            lab.append(np.argmax(predictions[i]))
        predictable = PrettyTable()
        predictable.add_column("filename",files)
        predictable.add_column("prediction",lab)
        print("\n\n\n predictions: \n\n")
        print(predictable)
    except:
        print("error predicting \n make sure model was trained properly")


print("\nloading.............100%\n\n\n")

#continous loop to keep it going
while (True):
    option = int(input("\n\n\n\nchoose option: \n \t1.train \n \t2.test \n \t3.predict \n \t4.exit \n\n\n"))
    if option == 1:
        print("getting ready to train data")
        train_data(int(input("\nenter no of epochs: ")))
        print("\n.....................training completed................")
        print("\n........................saved to file................")
    elif option==2:
        print("\nloading model.........\n")
        test()
    elif option==3:
        print("\nloading prediction model...........\n")
        predict()
    elif option==4:
        sys.exit(1)
    else:
        print("\ninvalid entry try again")
        continue
