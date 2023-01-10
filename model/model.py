import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Concatenate, Flatten, Add, Activation, Lambda
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from datetime import datetime


class PANOPTES:
    def __init__(self,
                 base_model_name='InceptionResNetV1',
                 auxiliary=False,
                 aux_weight=0.,
                 input_dim=(299, 299, 3),
                 feature_pool=True,
                 global_pool='avg',
                 covariate=None,   # number of covariates added
                 dropout=0.5,
                 n_classes=2,
                 saved_model=None):

        self.base_model_name = base_model_name
        self.auxiliary = auxiliary
        self.aux_weight = aux_weight
        self.input_dim = input_dim
        self.global_pool = global_pool
        self.feature_pool = feature_pool    
        self.covariate = covariate
        self.dropout = dropout    # dropout rate
        self.n_classes = n_classes
        self.classifier_history = None
        self.model = self.build()  # build the panoptes branches

        if saved_model:  # load weights from saved model if provided
            self.model.load_weights(saved_model)
            print('Loading saved model: ' + saved_model)    

        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

    def build(self):    # build the panoptes architecture, with options of softmax/acitvation outputs
        input_a = keras.Input(shape=self.input_dim, name='input_a')
        input_b = keras.Input(shape=self.input_dim, name='input_b')
        input_c = keras.Input(shape=self.input_dim, name='input_c')
        
        base_layers = __import__('layer',
                               fromlist=[self.base_model_name])  # legacy code, InceptionResNetV1, V2, w/o feature pooling
        
        print('Using base model: ' + self.base_model_name)
        base_model = getattr(base_layers, self.base_model_name)

        branch_a = base_model(input_shape=self.input_dim, dropout_rate=self.dropout, num_classes=self.n_classes)
        branch_a._name = self.base_model_name + '_a'
        for layer in branch_a.layers:
            layer._name = 'branch_a_' + layer.name

        branch_b = base_model(input_shape=self.input_dim, dropout_rate=self.dropout, num_classes=self.n_classes)
        branch_b._name = self.base_model_name + '_b'
        for layer in branch_b.layers:
            layer._name = 'branch_b_' + layer.name

        branch_c = base_model(input_shape=self.input_dim, dropout_rate=self.dropout, num_classes=self.n_classes)
        branch_c._name = self.base_model_name + '_c'
        for layer in branch_c.layers:
            layer._name = 'branch_c_' + layer.name
        
        xa, auxa = branch_a(input_a)
        xb, auxb = branch_b(input_b)
        xc, auxc = branch_c(input_c)
        
        x = Concatenate(axis=-1, name='conv_output')([xa, xb, xc])  # Output: 8 * 8 * 2688
        
        if self.feature_pool:    
            print('Feature pooling before final Dense layer.')
            x = Conv2D(2688, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same", name='feature_pool')(x)
        
        if self.global_pool == 'avg':    # global pooling options
            x = GlobalAveragePooling2D(name='avg_pool')(x) 
        elif self.global_pool == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)
        else:
            x = Flatten(name='flatten')(x)
        
        x = Dropout(self.dropout, name='latent_dropout')(x)

        if self.covariate is not None:
            input_d = keras.Input(shape=(self.covariate, ), name='input_d')    # adding covariate input
            xd = Dense(2, name='covariate_fc', activation="relu", kernel_regularizer=l2(0.0002))(input_d)
            x = Concatenate(axis=-1, name='covariate_concat')([x, xd])
            model_inputs = [input_a, input_b, input_c, input_d]
        
        else:
            model_inputs = [input_a, input_b, input_c]
        
        
        out = Dense(self.n_classes, name='prob', activation='softmax')(x)  # prob. output
        
        aux = Add(name='aux_added')([auxa, auxb, auxc])
        aux_out = Dense(self.n_classes, name='aux_prob', activation='softmax')(aux)
        
        if self.auxiliary:
            model_outputs = [out, aux_out]
        else:
            model_outputs = out

        panoptes_model = keras.Model(inputs=model_inputs,
                                     outputs=model_outputs, name='panoptes')
        
        print(panoptes_model.summary())

        print('Latent layer size: ' + str(panoptes_model.get_layer('latent_dropout').output.shape[1]))

        return panoptes_model


    def compile(self,
                loss_fn=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.AUC()],
                learning_rate=0.0001):
        if self.auxiliary:
            w = [1, self.aux_weight]
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                               loss=[loss_fn, loss_fn],
                               loss_weights=w,
                               metrics=metrics)

        else:
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                               loss=loss_fn,
                               metrics=metrics)

        
    def train(self,
              trn_data, val_data,
              n_epoch=10,
              steps=10000,
              patience=5,
              log_dir='./log',
              model_dir='./model',
              class_weight=None,
              ):
        
        print('Training patience: ' + str(patience))
        os.makedirs(log_dir, exist_ok=True)
        print('Training logs in: ' + log_dir)
        
        os.makedirs(model_dir, exist_ok=True)
        print('Saving model in: ' + model_dir)

        os.makedirs(model_dir + '/ckpt', exist_ok=True)

        csv_logger = tf.keras.callbacks.CSVLogger(log_dir + '/trn_history_logs.csv', append=True)

        tensor_board = tf.keras.callbacks.TensorBoard(log_dir + '/trn_tb_logs', update_freq=1000)
         
        ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + '/ckpt/weights.{epoch:03d}-{val_loss:.4f}.hdf5',
                                                  save_weights_only=True,
                                                  monitor='val_loss',
                                                  mode='min',
                                                  save_best_only=False)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          patience=patience,
                                                          restore_best_weights=True)

        self.classifier_history = self.model.fit(trn_data, validation_data=val_data,    # train step
                                                steps_per_epoch=steps,
                                                epochs=n_epoch,
                                                callbacks=[csv_logger, tensor_board, ckpt, early_stopping])
            
        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
        
        self.model.save_weights(model_dir + "/panoptes_weights_final.h5")
        print('Final model saved.')

    def inference(self, x, batch_size=8):
        inputs = self.model.input
        activation = self.model.get_layer('latent_dropout').output
        prob = self.model.get_layer('prob').output
        inference_model = keras.Model(inputs=inputs,
                                      outputs=[activation, prob])
        if isinstance(x, np.ndarray):    # if input is numpy array
            res = inference_model.predict(x, batch_size=batch_size)
        else:    # if input is tf dataset
            res = inference_model.predict(x)
        return res

    def print_attr(self):
        print('Model attributes:')
        print('Base model name: ' + str(self.base_model_name))
        print('Auxiliary output: ' + str(self.auxiliary))
        print('Covariate: ' + str(self.covariate))
        print('Input size: ' + str(self.input_dim))
        print('Global pooling: ' + str(self.global_pool))
        print('Feature pooling:' + str(self.feature_pool))
        print('Number of outcome classes: ' + str(self.n_classes))
        print('Dropout: ' + str(self.dropout))
        print('Model date-time: ' + str(self.datetime))


