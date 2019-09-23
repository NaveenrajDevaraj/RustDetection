from model import *
from data import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(rotation_range=0,
                    width_shift_range=0,
                    height_shift_range=0,
                    shear_range=0,
                    zoom_range=0,
                    horizontal_flip=False,
                    fill_mode='nearest')
#myGene = trainGenerator(2,'data','images','annotations',data_gen_args,save_to_dir = None)
myGene = trainGenerator(2,'mydata','Images','Annotations',data_gen_args,save_to_dir = None)

model = unet()

early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
model_checkpoint = ModelCheckpoint('trained_weights/unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=10,callbacks=[model_checkpoint, early_stopping])