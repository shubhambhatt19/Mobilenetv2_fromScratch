from data import parsed_data
import tensorflow as tf
from tensorflow.keras import Model
import config as cfg
from mobilenetv2 import mobilenetv2



ds_train, ds_test = parsed_data(train_record_path = cfg.INPUT_PATH+"/train.record", test_record_path = cfg.INPUT_PATH+"/train.record")

ds_train = ds_train.batch(cfg.BATCH_SIZE)
ds_test = ds_test.batch(cfg.BATCH_SIZE)


inputs = tf.keras.Input(shape = cfg.inp_image_size, batch_size = cfg.BATCH_SIZE)

model = mobilenetv2(inputs=inputs, noOfClass=cfg.NO_OF_CLASSES)

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()

model.fit(ds_train,epochs=1,verbose=1,steps_per_epoch=cfg.EPOCH)

model.evaluate(ds_test)