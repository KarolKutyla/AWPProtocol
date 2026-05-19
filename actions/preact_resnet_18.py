import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------
# PreAct Residual Block
# ---------------------------
def preact_resnet_block(x, filters, stride=1, width_mult=1):
    bn1 = layers.BatchNormalization()(x)
    act1 = layers.Activation("relu")(bn1)

    shortcut = x

    # scaled filters
    f = int(filters * width_mult)

    # first conv
    y = layers.Conv2D(f, 3, stride, padding="same", use_bias=False)(act1)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)

    # second conv
    y = layers.Conv2D(f, 3, 1, padding="same", use_bias=False)(y)

    # shortcut projection if needed
    if stride != 1 or x.shape[-1] != f:
        shortcut = layers.Conv2D(f, 1, stride, use_bias=False)(act1)

    out = layers.Add()([y, shortcut])
    return out


# ---------------------------
# Stage (stack of blocks)
# ---------------------------
def make_stage(x, filters, blocks, stride, width_mult):
    x = preact_resnet_block(x, filters, stride=stride, width_mult=width_mult)
    for _ in range(1, blocks):
        x = preact_resnet_block(x, filters, stride=1, width_mult=width_mult)
    return x


# ---------------------------
# PreActResNet18
# ---------------------------
def PreActResNet18(input_shape=(32, 32, 3), num_classes=10, width_mult=10):
    inputs = keras.Input(shape=input_shape)

    # initial conv
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)

    # ResNet18 structure
    x = make_stage(x, 64, 2, stride=1, width_mult=width_mult)
    x = make_stage(x, 128, 2, stride=2, width_mult=width_mult)
    x = make_stage(x, 256, 2, stride=2, width_mult=width_mult)
    x = make_stage(x, 512, 2, stride=2, width_mult=width_mult)

    # head
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs, name=f"PreActResNet18_w{width_mult}")