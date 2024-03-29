import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.regularizers import L2


def resnet_v1_stem(input, l2=0.0002):
    # The stem of the Inception-ResNet-v1 network.
    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = Conv2D(32, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2), padding="same")(
        input)  # 149 * 149 * 32
    x = Conv2D(32, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(x)  # 147 * 147 * 32
    x = Conv2D(64, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(x)  # 147 * 147 * 64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)  # 73 * 73 * 64
    x = Conv2D(80, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(x)  # 73 * 73 * 80
    x = Conv2D(192, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(x)  # 71 * 71 * 192
    x = Conv2D(256, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2), padding="same")(
        x)  # 35 * 35 * 256
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    return x


def inception_resnet_v1_A(input, scale_residual=True, l2=0.0002):
    # Architecture of Inception_ResNet_A block which is a 35 * 35 grid module.
    ar1 = Conv2D(32, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    ar2 = Conv2D(32, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    ar2 = Conv2D(32, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(ar2)
    ar3 = Conv2D(32, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    ar3 = Conv2D(32, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(ar3)
    ar3 = Conv2D(32, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(ar3)
    merged = concatenate([ar1, ar2, ar3], axis=3)
    ar = Conv2D(256, (1, 1), activation="linear", padding="same")(merged)
    if scale_residual: ar = Lambda(lambda a: a * 0.1)(ar)
    output = add([input, ar])
    output = BatchNormalization(axis=3)(output)
    output = Activation("relu")(output)
    return output


def inception_resnet_v1_B(input, scale_residual=True, l2=0.0002):
    # Architecture of Inception_ResNet_B block which is a 17 * 17 grid module.
    br1 = Conv2D(128, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    br2 = Conv2D(128, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    br2 = Conv2D(128, (1, 7), kernel_regularizer=L2(l2), activation="relu", padding="same")(br2)
    br2 = Conv2D(128, (7, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(br2)
    merged = concatenate([br1, br2], axis=3)
    br = Conv2D(896, (1, 1), activation="linear", padding="same")(merged)
    if scale_residual: br = Lambda(lambda b: b * 0.1)(br)
    output = add([input, br])
    output = BatchNormalization(axis=3)(output)
    output = Activation("relu")(output)
    return output


def inception_resnet_v1_C(input, scale_residual=True, l2=0.0002):
    # Architecture of Inception_ResNet_C block which is a 8 * 8 grid module.
    cr1 = Conv2D(192, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    cr2 = Conv2D(192, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    cr2 = Conv2D(192, (1, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(cr2)
    cr2 = Conv2D(192, (3, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(cr2)
    merged = concatenate([cr1, cr2], axis=3)
    cr = Conv2D(1792, (1, 1), activation="linear", padding="same")(merged)
    if scale_residual: cr = Lambda(lambda c: c * 0.1)(cr)
    output = add([input, cr])
    output = BatchNormalization(axis=3)(output)
    output = Activation("relu")(output)
    return output


def reduction_resnet_A(input, k=192, l=224, m=256, n=384, l2=0.0002):
    # Architecture of a 35 * 35 to 17 * 17 Reduction_ResNet_A block. It is used by both v1 and v2 Inception-ResNets.
    rar1 = MaxPooling2D((3, 3), strides=(2, 2))(input)
    rar2 = Conv2D(n, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(input)
    rar3 = Conv2D(k, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    rar3 = Conv2D(l, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(rar3)
    rar3 = Conv2D(m, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(rar3)
    merged = concatenate([rar1, rar2, rar3], axis=3)
    rar = BatchNormalization(axis=3)(merged)
    rar = Activation("relu")(rar)
    return rar


def reduction_resnet_v1_B(input, l2=0.0002):
    # Architecture of a 17 * 17 to 8 * 8 Reduction_ResNet_B block.
    rbr1 = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(input)
    rbr2 = Conv2D(256, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    rbr2 = Conv2D(384, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(rbr2)
    rbr3 = Conv2D(256, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    rbr3 = Conv2D(256, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(rbr3)
    rbr4 = Conv2D(256, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    rbr4 = Conv2D(256, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(rbr4)
    rbr4 = Conv2D(256, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(rbr4)
    merged = concatenate([rbr1, rbr2, rbr3, rbr4], axis=3)
    rbr = BatchNormalization(axis=3)(merged)
    rbr = Activation("relu")(rbr)
    return rbr

"""
Inception Resnet V2 blocks
"""
def resnet_v2_stem(input, l2=0.0002):
    # The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.

    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = Conv2D(32, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(input)  # 149 * 149 * 32
    x = Conv2D(32, (3, 3), kernel_regularizer=L2(l2), activation="relu")(x)  # 147 * 147 * 32
    x = Conv2D(64, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(x)  # 147 * 147 * 64

    x1 = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x2 = Conv2D(96, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(x)

    x = concatenate([x1, x2], axis=3)  # 73 * 73 * 160

    x1 = Conv2D(64, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(x)
    x1 = Conv2D(96, (3, 3), kernel_regularizer=L2(l2), activation="relu")(x1)

    x2 = Conv2D(64, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(x)
    x2 = Conv2D(64, (7, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(x2)
    x2 = Conv2D(64, (1, 7), kernel_regularizer=L2(l2), activation="relu", padding="same")(x2)
    x2 = Conv2D(96, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="valid")(x2)

    x = concatenate([x1, x2], axis=3)  # 71 * 71 * 192

    x1 = Conv2D(192, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(x)

    x2 = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = concatenate([x1, x2], axis=3)  # 35 * 35 * 384

    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    return x


def inception_resnet_v2_A(input, scale_residual=True, l2=0.0002):
    # Architecture of Inception_ResNet_A block which is a 35 * 35 grid module.

    ar1 = Conv2D(32, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)

    ar2 = Conv2D(32, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    ar2 = Conv2D(32, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(ar2)

    ar3 = Conv2D(32, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    ar3 = Conv2D(48, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(ar3)
    ar3 = Conv2D(64, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(ar3)

    merged = concatenate([ar1, ar2, ar3], axis=3)

    ar = Conv2D(384, (1, 1), kernel_regularizer=L2(l2), activation="linear", padding="same")(merged)
    if scale_residual: ar = Lambda(lambda a: a * 0.1)(ar)

    output = add([input, ar])
    output = BatchNormalization(axis=3)(output)
    output = Activation("relu")(output)

    return output


def inception_resnet_v2_B(input, scale_residual=True, l2=0.0002):
    # Architecture of Inception_ResNet_B block which is a 17 * 17 grid module.

    br1 = Conv2D(192, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)

    br2 = Conv2D(128, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    br2 = Conv2D(160, (1, 7), kernel_regularizer=L2(l2), activation="relu", padding="same")(br2)
    br2 = Conv2D(192, (7, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(br2)

    merged = concatenate([br1, br2], axis=3)

    br = Conv2D(1152, (1, 1), kernel_regularizer=L2(l2), activation="linear", padding="same")(merged)
    if scale_residual: br = Lambda(lambda b: b * 0.1)(br)

    output = add([input, br])
    output = BatchNormalization(axis=3)(output)
    output = Activation("relu")(output)

    return output


def inception_resnet_v2_C(input, scale_residual=True, l2=0.0002):
    # Architecture of Inception_ResNet_C block which is a 8 * 8 grid module.

    cr1 = Conv2D(192, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)

    cr2 = Conv2D(192, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    cr2 = Conv2D(224, (1, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(cr2)
    cr2 = Conv2D(256, (3, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(cr2)

    merged = concatenate([cr1, cr2], axis=3)

    cr = Conv2D(2144, (1, 1), kernel_regularizer=L2(l2), activation="linear", padding="same")(merged)
    if scale_residual: cr = Lambda(lambda c: c * 0.1)(cr)

    output = add([input, cr])
    output = BatchNormalization(axis=3)(output)
    output = Activation("relu")(output)

    return output


def reduction_resnet_v2_B(input, l2=0.0002):
    # Architecture of a 17 * 17 to 8 * 8 Reduction_ResNet_B block.

    rbr1 = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(input)

    rbr2 = Conv2D(256, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    rbr2 = Conv2D(384, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(rbr2)

    rbr3 = Conv2D(256, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    rbr3 = Conv2D(288, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(rbr3)

    rbr4 = Conv2D(256, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(input)
    rbr4 = Conv2D(288, (3, 3), kernel_regularizer=L2(l2), activation="relu", padding="same")(rbr4)
    rbr4 = Conv2D(320, (3, 3), kernel_regularizer=L2(l2), activation="relu", strides=(2, 2))(rbr4)

    merged = concatenate([rbr1, rbr2, rbr3, rbr4], axis=3)
    rbr = BatchNormalization(axis=3)(merged)
    rbr = Activation("relu")(rbr)

    return rbr

    
"""
Backbone of InceptionResNetV1 model.
Default Input: (299, 299, 3)
Default Ouput: (8, 8, 896)
"""
def InceptionResNetV1(input_shape, dropout_rate=0.8, num_classes=1000, l2=0.0002):
    # Input shape is 299 * 299 * 3
    input = tf.keras.Input(shape=input_shape)
    x = resnet_v1_stem(input, l2=l2)  # Output: 35 * 35 * 256
    
    # 5 x Inception A
    for i in range(5):
        x = inception_resnet_v1_A(x, l2=l2)
        # Output: 35 * 35 * 256
    
    # Reduction A
    x = reduction_resnet_A(x, k=192, l=192, m=256, n=384, l2=l2)  # Output: 17 * 17 * 896
    
    # 10 x Inception B
    for i in range(10):
        x = inception_resnet_v1_B(x, l2=l2)
        # Output: 17 * 17 * 896

    # auxiliary
    loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(x)
    loss2_conv_a = Conv2D(128, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(loss2_ave_pool)
    loss2_conv_b = Conv2D(768, (5, 5), kernel_regularizer=L2(l2), activation="relu", padding="same")(loss2_conv_a)
    loss2_conv_b = BatchNormalization(axis=3)(loss2_conv_b)
    loss2_conv_b = Activation('relu')(loss2_conv_b)
    loss2_flat = Flatten()(loss2_conv_b)
    loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=L2(l2))(loss2_flat)
    loss2_drop_fc = Dropout(dropout_rate)(loss2_fc)    # using keras.model.fit, set training automatically
    loss2_classifier = Dense(num_classes, name='loss2/classifier', kernel_regularizer=L2(l2))(loss2_drop_fc)
    
    # Reduction B
    x = reduction_resnet_v1_B(x, l2=l2)  # Output: 8 * 8 * 1792
    
    # 5 x Inception C
    for i in range(5):
        x = inception_resnet_v1_C(x, l2=l2)
        # Output: 8 * 8 * 1792
    x = Conv2D(896, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(x)

    model = tf.keras.Model(inputs=input, outputs=[x, loss2_classifier], name='InceptionResNetV1')
    
    return model

"""
Backbone of InceptionResNetV2 model.
Default Input: (299, 299, 3)
Default Ouput: (8, 8, 896)
"""
def InceptionResNetV2(input_shape, dropout_rate=0.8, num_classes=1000, l2=0.0):    # 0.0002
    # Input shape is 299 * 299 * 3
    input = tf.keras.Input(shape=input_shape)
    x = resnet_v2_stem(input)  # Output: 35 * 35 * 256

    # 5 x Inception A
    for i in range(5):
        x = inception_resnet_v2_A(x, l2=l2)
        # Output: 35 * 35 * 256

    # Reduction A
    x = reduction_resnet_A(x, k=256, l=256, m=384, n=384, l2=l2)  # Output: 17 * 17 * 896

    # 10 x Inception B
    for i in range(10):
        x = inception_resnet_v2_B(x, l2=l2)
        # Output: 17 * 17 * 896

    # auxiliary
    loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(x)

    loss2_conv_a = Conv2D(128, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(
        loss2_ave_pool)
    loss2_conv_b = Conv2D(768, (5, 5), kernel_regularizer=L2(l2), activation="relu", padding="same")(
        loss2_conv_a)

    loss2_conv_b = BatchNormalization(axis=3)(loss2_conv_b)

    loss2_conv_b = Activation('relu')(loss2_conv_b)

    loss2_flat = Flatten()(loss2_conv_b)

    loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=L2(l2))(loss2_flat)

    loss2_drop_fc = Dropout(dropout_rate)(loss2_fc)    # using keras.model.fit, set training automatically

    loss2_classifier = Dense(num_classes, name='loss2/classifier', kernel_regularizer=L2(l2))(loss2_drop_fc)

    # Reduction B
    x = reduction_resnet_v2_B(x, l2=l2)  # Output: 8 * 8 * 1792

    # 5 x Inception C
    for i in range(5):
        x = inception_resnet_v2_C(x, l2=l2)
        # Output: 8 * 8 * 1792

    x = Conv2D(896, (1, 1), kernel_regularizer=L2(l2), activation="relu", padding="same")(x)

    model = tf.keras.Model(inputs=input, outputs=[x, loss2_classifier], name='InceptionResNetV2')

    return model
    

