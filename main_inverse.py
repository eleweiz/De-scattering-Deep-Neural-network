from model import *
from keras.utils import plot_model
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio 
from Generator_data_function import DataGenerator
H=64   # replace the x dimension of your own data
W=64   # replace the y dimension of your own data
D=64   # replace the z dimension of your own data

matfn = './Training_SpineN_in.mat'
data= h5py.File(matfn)
trainX = data['I_in']
trainX=np.reshape(trainX,trainX.shape + (1,)) 
data=None
matfn1 = './Training_SpineN_out.mat'
data1= h5py.File(matfn1)
trainY = data1['I_out']
data1=None
trainY=np.reshape(trainY,trainY.shape + (1,))
print(np.isnan(trainX).any())
print(np.isnan(trainY).any()) 
portion=0.9
tem=np.size(trainX,0)
Train=np.floor(tem*portion)
a_train=np.arange(0, Train, 1, int)
a_val=np.arange(Train,tem, 1, int)

#######  Generator
# Parameters
params = {'dim': (H,W,D),
          'batch_size': 3,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator      = DataGenerator(a_train, trainX, trainY, **params)
validation_generator    = DataGenerator(a_val, trainX, trainY, **params)


model = unet(input_size = (H,W,D,1), learning_rate = 1e-4)
plot_model(model, to_file='model_fig.png')
filepath="Trained-Model-{epoch:02d}-{val_loss:.2f}.hdf5"
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True, period=1)
# period, the distance of epochs for saving points
'''
############ train
# '''
history =model.fit_generator(training_generator,steps_per_epoch=5,epochs=2,callbacks=[model_checkpoint],validation_data=validation_generator, validation_steps=3)
 
# steps_per_epoch: typically be equal to ceil(num_samples / batch_size)
# validation_steps: Only relevant if validation_data is a generator. Total number of steps (batches of samples) to yield from validation_data generator before stopping at the end of every epoch. It should typically be equal to the number of samples of your validation dataset divided by the batch size.
# An epoch finishes when steps_per_epoch batches have been seen by the model.
# visulization loss
# Plot training & validation accuracy values

dataNew = './Results/loss_history.mat'    # save data
Results_CNN=history.history
sio.savemat(dataNew, history.history) 
