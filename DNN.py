#OG ssn
import tensorflow as tf
import h5py
import keras.backend as K
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot, Flatten,Dropout
from keras.models import Sequential, load_model,Model


parser = argparse.ArgumentParser()
parser.add_argument('--onlyPlot', action="store_true", dest="onlyPlot") #defaults to False
parser.add_argument('--small', action = "store_true", dest = "small")

args = parser.parse_args()


print ('Extracting')

if args.small==False:
    f = h5py.File('data_HWWqqqq.hdf5', 'r')
    jet_data = f['updated jet attributes'][:]
    labels = f['jet labels'][:]
else:
    f = h5py.File('data_HWWqqqq.hdf5', 'r')
    jet_data = f['updated jet attributes'][:500001]
    labels = f['jet labels'][:500001]


jet_data = np.nan_to_num(jet_data, nan = -1)



#set controllable values


eventDataLength = 10


trainingDataLength = int(len(jet_data)*0.6)

validationDataLength = int(len(jet_data)*0.2)






numberOfEpochs = 100
batchSize = 1024
modelName = "test_model"

#Creating training data

print ("Preparing Data")

tau21 = jet_data[:,5]/jet_data[:,4]


tau21 = np.nan_to_num(tau21, nan = -1)



tau32 = jet_data[:,6]/jet_data[:,5]


tau32 = np.nan_to_num(tau32, nan = 999)


tau43 = jet_data[:,7]/jet_data[:,6]

tau43 = np.nan_to_num(tau43, nan = -1)




additional_features = np.vstack((tau21, tau32, tau43))


jet_data = np.concatenate((jet_data, additional_features.T), axis = 1)

jet_mass = jet_data[:,3]
print (jet_mass.shape)
jet_data = np.delete(jet_data,3,axis=1)

print(jet_data.shape)

trainingLabels = np.array(labels[0:trainingDataLength])
TrainingJetInfo = np.array(jet_data[0:trainingDataLength,0:13])

for i in TrainingJetInfo[:20]:
    print(i[:4])

print(trainingLabels[:10])



validationLabels = np.array(labels[trainingDataLength:trainingDataLength + validationDataLength])
ValidationJetInfo = np.array(jet_data[trainingDataLength:trainingDataLength + validationDataLength, 0:13])

print(validationLabels[:10])

testLabels = np.array(labels[trainingDataLength+validationDataLength:])
jetTestData = np.array(jet_data[trainingDataLength+validationDataLength:])
TestJetInfo = np.array(jet_data[trainingDataLength+validationDataLength:, 0:13])
jet_mass = np.array(jet_mass[trainingDataLength+validationDataLength:])


inputJet = Input(shape= (13,), name = "inputJet")

denseEndOne = Dense(64, activation="relu", name="denseEndOne")(inputJet)
normEndOne = BatchNormalization(momentum=0.8, name="normEndOne")(denseEndOne)
denseEndTwo = Dense(32, activation="relu", name="denseEndTwo")(normEndOne)
denseEndThree = Dense(32, activation="relu", name="denseEndThree")(denseEndTwo)
output = Dense(3, activation="softmax", name="output")(denseEndThree) ####Made a change to the activation function, and the number of neurons in the output layer from 1 to 3

print("Compiling")

model = Model(inputs=[inputJet], outputs=[output])

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc']) ###Made a change to the loss function


print(model.summary())
if args.onlyPlot == False:
    print('Calculating')

    modelCallbacks = [EarlyStopping(patience=10),
                    ModelCheckpoint(filepath="./data/"+modelName+".h5", save_weights_only=True,
                                    save_best_only=True)]

    history = model.fit([TrainingJetInfo], trainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                        callbacks=modelCallbacks,
                        validation_data=([ValidationJetInfo], validationLabels))

    with open("./data/"+modelName+",history.json", "w") as f:
        json.dump(history.history,f)

    print("Loading weights")

    model.load_weights("./data/"+modelName+".h5")

    model.save("./data/"+modelName+",model")

    print("Predicting")
    predictions = model.predict(TestJetInfo)
    print (predictions[:10])
    print(testLabels[:10])
    data_file = h5py.File("snn10_data.hdf5", "w")
    dset1 = data_file.create_dataset("predictions", data= predictions)
    dset2 = data_file.create_dataset("testLabels", data=testLabels)
    dset3 = data_file.create_dataset('jetTestData', data = jetTestData)
    dset4 = data_file.create_dataset("jet_mass", data =jet_mass)





#confusion matrix
    recoil_pred = 0
    threeq_pred = 0
    fourq_pred = 0
    confusion_matrix = np.zeros((3,3))
    for i in range(len(testLabels)):
        #confusion matrix and efficiency curves

        if predictions[i].max() == predictions[i][0]:
            recoil_pred+=1
          
            if testLabels[i].tolist() == [1,0,0]:
                confusion_matrix[0,0]+=1
 
            elif testLabels[i].tolist()==[0,1,0]:
                confusion_matrix[1,0]+=1
            elif testLabels[i].tolist()==[0,0,1]:
                confusion_matrix[2,0]+=1

            
        elif predictions[i].max() == predictions[i][1]:
            threeq_pred+=1
           
            if testLabels[i].tolist() == [1,0,0]:
                confusion_matrix[0,1]+=1
            elif testLabels[i].tolist()==[0,1,0]:
                confusion_matrix[1,1]+=1
             
            elif testLabels[i].tolist()==[0,0,1]:
                confusion_matrix[2,1]+=1

            
        elif predictions[i].max() ==predictions[i][2]:
            fourq_pred +=1
            
            if testLabels[i].tolist() == [1,0,0]:
                confusion_matrix[0,2]+=1
            elif testLabels[i].tolist()==[0,1,0]:
                confusion_matrix[1,2]+=1
            elif testLabels[i].tolist()==[0,0,1]:
                confusion_matrix[2,2]+=1
             





#ROC curve
    true_label_3q =[]
    scores_3q_vs_recoil = []
    true_label_4q = []
    scores_4q_vs_recoil = []
    true_label_h_vs_r = []
    scores_h_vs_r = []
    for i in range(len(testLabels)):
        scores_h_vs_r.append((1-predictions[i][0]))
        true_label_h_vs_r.append((1-testLabels[i][0]))
        if not np.isnan(predictions[i][1]/(predictions[i][0]+predictions[i][1])):
            scores_3q_vs_recoil.append(predictions[i][1]/(predictions[i][0]+predictions[i][1]))
            true_label_3q.append(testLabels[i][1])
        if not np.isnan(predictions[i][2]/(predictions[i][0]+predictions[i][2])):
            scores_4q_vs_recoil.append(predictions[i][2]/(predictions[i][0]+predictions[i][2]))
            true_label_4q.append(testLabels[i][2])



        
        
    
    
    fpr1, tpr1, thresholds = roc_curve(true_label_3q,scores_3q_vs_recoil)
    fpr2, tpr2, thresholds = roc_curve(true_label_4q,scores_4q_vs_recoil)
    fpr4, tpr4, thresholds = roc_curve(true_label_h_vs_r, scores_h_vs_r)


    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, label='three quark vs recoil')
    plt.plot(fpr2, tpr2, label='four quark vs recoil')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('ROC.png')

    #confusion matrix
    confusion_matrix = confusion_matrix.astype(np.float64)
    confusion_matrix[:,0]/=recoil_pred
    if threeq_pred !=0:
        confusion_matrix[:,1]/=threeq_pred
    else:
        confusion_matrix[:,1] = 0
    if fourq_pred != 0:
        confusion_matrix[:,2]/=fourq_pred
    else:
        confusion_matrix[:,2] = 0
    
    
    confusion_matrix= np.round(confusion_matrix,decimals=3)

    fig, ax = plt.subplots()

    #Plot the confusion matrix
    ax.imshow(confusion_matrix, cmap='Blues')

    # Add a colorbar
    cbar = ax.figure.colorbar(ax.imshow(confusion_matrix, cmap='Blues'), ax=ax)

    # Add labels to the x-axis and y-axis
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    # Add tick marks and labels
    ax.set_xticks(np.arange(confusion_matrix.shape[1]))
    ax.set_yticks(np.arange(confusion_matrix.shape[0]))
    ax.set_xticklabels(['Recoil', 'Three Quark Higgs','Four Quark Higgs'])  # Replace with your class labels
    ax.set_yticklabels(['Recoil', 'Three Quark Higgs','Four Quark Higgs'])  # Replace with your class labels

    # Set the alignment of the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

    #Display the confusion matrix
    plt.savefig('confusion_matrix.png')
