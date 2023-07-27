##GNN using normalized jet constituents data on top of jet four vectors
import tensorflow as tf
import h5py
import keras.backend as K
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, BatchNormalization, Conv1D, Lambda
from keras.models import Model


parser = argparse.ArgumentParser()

parser.add_argument('--small', action = "store_true", dest = "small")

args = parser.parse_args()




print ('Extracting')

if args.small==False:
    f = h5py.File('data_HWWqqqq.hdf5', 'r')
    fjc_data = f["jet components"][:]
    jet_data = f['updated jet attributes'][:]
    labels = f['jet labels'][:]
    fjc_normal = f["normalized jet components"][:]

else:

    f = h5py.File('data_HWWqqqq.hdf5', 'r')
    fjc_data = f["jet components"][:500001]
    jet_data = f['updated jet attributes'][:500001]
    labels = f['jet labels'][:500001]
    fjc_normal = f["normalized jet components"][:500001]



#Set controllable variables
particlesConsidered = 30
entriesPerParticle = 4
eventDataLength = 10
trainingDataLength = int(len(fjc_normal)*0.6)
validationDataLength = int(len(fjc_normal)*0.2)
numberOfEpochs = 100
batchSize = 1024
modelName = "test_model"

#Creating training data
print ("Preparing Data")
particleTrainingData = np.array(fjc_normal[0:trainingDataLength,])
trainingLabels = np.array(labels[0:trainingDataLength])
TrainingJetInfo = np.array(jet_data[0:trainingDataLength,0:3])


particleValidationData = np.array(fjc_normal[trainingDataLength:trainingDataLength+validationDataLength,])
validationLabels = np.array(labels[trainingDataLength:trainingDataLength + validationDataLength])
ValidationJetInfo = np.array(jet_data[trainingDataLength:trainingDataLength + validationDataLength, 0:3])


particleTestData = np.array(fjc_normal[trainingDataLength+validationDataLength:,])
testLabels = np.array(labels[trainingDataLength + validationDataLength:])
jetTestData = np.array(jet_data[trainingDataLength+validationDataLength:])
TestJetInfo = np.array(jet_data[trainingDataLength+validationDataLength:, 0:3])



#Defining the receiving matrix for particles

RR = []
for i in range(particlesConsidered):
    row = []
    for j in range(particlesConsidered * (particlesConsidered - 1)):
        if j in range(i * (particlesConsidered - 1), (i + 1) * (particlesConsidered - 1)):
            row.append(1.0)
        else:
            row.append(0.0)
    RR.append(row)
RR = np.array(RR)
RR = np.float32(RR)
RRT = np.transpose(RR)

# Defining the sending matrix for particles

RST = []
for i in range(particlesConsidered):
    for j in range(particlesConsidered):
        row = []
        for k in range(particlesConsidered):
            if k == j:
                row.append(1.0)
            else:
                row.append(0.0)
        RST.append(row)
rowsToRemove = []
for i in range(particlesConsidered):
    rowsToRemove.append(i * (particlesConsidered + 1))
RST = np.array(RST)
RST = np.float32(RST)
RST = np.delete(RST, rowsToRemove, 0)
RS = np.transpose(RST)



# Creates and trains the neural net

# Particle data interaction NN
inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")
inputJet = Input(shape = (3), name ="inputJet")


XdotRR = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RR, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRR")(inputParticle)

XdotRS = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RS, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRS")(inputParticle)

Bpp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bpp")([XdotRR, XdotRS])


convNormOne = BatchNormalization(momentum=0.6, name="convNormOne")(Bpp)
convOneParticle = Conv1D(64, kernel_size=1, activation="relu", name="convOneParticle")(convNormOne)
convTwoParticle = Conv1D(64, kernel_size=1, activation="relu", name="convTwoParticle")(convOneParticle)
convThreeParticle = Conv1D(64, kernel_size=1, activation="relu", name="convThreeParticle")(convTwoParticle)

Epp = BatchNormalization(momentum=0.6, name="Epp")(convThreeParticle)


# Combined prediction NN
EppBar = Lambda(lambda listOfTensors: tf.transpose(tf.matmul(tf.transpose(listOfTensors[0], perm=(0, 2, 1)), tf.multiply(tf.expand_dims(tf.repeat(tf.cast(listOfTensors[1][:,1:,0]>0., listOfTensors[0].dtype), particlesConsidered, axis=1), axis=-1), np.expand_dims(RRT, axis=0))),
                                                perm=(0, 2, 1)), name="EppBar")([Epp, inputParticle])
C = Lambda(lambda listOfTensors: tf.concat((listOfTensors[0], listOfTensors[1]), axis=2), name="C")(
    [inputParticle, EppBar])

convPredictOne = Conv1D(32, kernel_size=1, activation="relu", name="convPredictOne")(C)
convPredictTwo = Conv1D(32, kernel_size=1, activation="relu", name="convPredictTwo")(convPredictOne)
convPredictThree = Conv1D(32, kernel_size=1, activation="relu", name="convPredictThree")(convPredictTwo)


O = Conv1D(24, kernel_size=1, activation="relu", name="O")(convPredictThree)

# Calculate output
OBar = Lambda(lambda listOfTensors: K.sum(tf.multiply(listOfTensors[0], tf.expand_dims(tf.cast(listOfTensors[1][:,:,0]>0., listOfTensors[0].dtype), axis=-1)), axis=1), name="OBar")([O, inputParticle])
combJet = tf.keras.layers.Concatenate(axis =1)([OBar, inputJet])
denseEndOne = Dense(60, activation="relu", name="denseEndOne")(combJet)
normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
denseEndTwo = Dense(30, activation="relu", name="denseEndTwo")(normEndOne)
denseEndThree = Dense(20, activation="relu", name="denseEndThree")(denseEndTwo)
output = Dense(3, activation="softmax", name="output")(denseEndThree) 
print("Compiling")

model = Model(inputs=[inputParticle, inputJet], outputs=[output])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) 
print(model.summary())

print('Calculating')

modelCallbacks = [EarlyStopping(patience=10),
                ModelCheckpoint(filepath="./data/"+modelName+".h5", save_weights_only=True,
                                save_best_only=True)]

history = model.fit([particleTrainingData, TrainingJetInfo], trainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                    callbacks=modelCallbacks,
                    validation_data=([particleValidationData, ValidationJetInfo], validationLabels))

with open("./data/"+modelName+",history.json", "w") as f:
    json.dump(history.history,f)

print("Loading weights")



model.load_weights("./data/"+modelName+".h5")

model.save("./data/"+modelName+",model")

    



        


print("Predicting")
predictions = model.predict([particleTestData, TestJetInfo])
#save predictions to a hp5y file
data_file = h5py.File("gnn_data.hdf5", "w")
dset1 = data_file.create_dataset("predictions", data= predictions)
dset2 = data_file.create_dataset("testLabels", data=testLabels)
dset3 = data_file.create_dataset('jetTestData', data = jetTestData)

#Generating plots to evaluate model performance

#confusion matrix
recoil_pred = 0
threeq_pred = 0
fourq_pred = 0
confusion_matrix = np.zeros((3,3))
for i in range(len(testLabels)):
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
    if not np.isnan(predictions[i][1]/(predictions[i][0]+predictions[i][1])): #remove nan values
        scores_3q_vs_recoil.append(predictions[i][1]/(predictions[i][0]+predictions[i][1]))
        true_label_3q.append(testLabels[i][1])
    if not np.isnan(predictions[i][2]/(predictions[i][0]+predictions[i][2])): #remove nan values
        scores_4q_vs_recoil.append(predictions[i][2]/(predictions[i][0]+predictions[i][2]))
        true_label_4q.append(testLabels[i][2])
        
fpr1, tpr1, thresholds = roc_curve(true_label_3q,scores_3q_vs_recoil)
fpr2, tpr2, thresholds = roc_curve(true_label_4q,scores_4q_vs_recoil)
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

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

#Display the confusion matrix
plt.savefig('confusion_matrix.png')
