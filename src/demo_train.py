import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To control the verbosity of TensorFlow messages

import tensorflow as tf
print('\ntensorflow version : ', tf.__version__)

from tensorflow.keras.applications import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K

import copy
import cv2, os
import numpy as np
import imgaug.augmenters as iaa

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BIN, OVERLAP = 6, 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car', 'Truck', 'Van', 'Tram']
BATCH_SIZE = 8
AUGMENTATION = False

## Select model and input size
select_model = 'resnet50'  # Options: resnet18, resnet50, resnet101, resnet152, vgg11, vgg16, vgg19, efficientnetb0, efficientnetb5, mobilenetv2

label_dir = '../dataset/kitti/training/label_2/'
image_dir = '../dataset/kitti/training/image_2/'

seq = iaa.Sequential([
    iaa.Crop(px=(0, 7)),  # Crop between 0 to 7 pixels from the left side
    iaa.Crop(px=(7, 0)),  # Crop between 0 to 7 pixels from right to left
    iaa.GaussianBlur(sigma=(0, 3.0))  
])

###### PreProcessing #####
def compute_anchors(angle):
    anchors = []
    wedge = 2.*np.pi/BIN

    l_index = int(angle/wedge)
    r_index = l_index + 1
    
    if (angle - l_index*wedge) < wedge/2 * (1+OVERLAP/2):
        anchors.append([l_index, angle - l_index*wedge])
    if (r_index*wedge - angle) < wedge/2 * (1+OVERLAP/2):
        anchors.append([r_index%BIN, angle - r_index*wedge])
    return anchors

def parse_annotation(label_dir, image_dir):
    all_objs = []
    dims_avg = {key: np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key: 0 for key in VEHICLES}
        
    for label_file in os.listdir(label_dir):
        image_file = label_file.replace('txt', 'png')

        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded  = np.abs(float(line[2]))

            if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi/2.
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.*np.pi
                new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi)

                obj = {'name': line[0],
                       'image': image_file,
                       'xmin': int(float(line[4])),
                       'ymin': int(float(line[5])),
                       'xmax': int(float(line[6])),
                       'ymax': int(float(line[7])),
                       'dims': np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                      }
                
                dims_avg[obj['name']]  = dims_cnt[obj['name']]*dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                all_objs.append(obj)
            
    return all_objs, dims_avg

all_objs, dims_avg = parse_annotation(label_dir, image_dir)

for obj in all_objs:
    # Fix dimensions
    obj['dims'] = obj['dims'] - dims_avg[obj['name']]
    
    # Fix orientation and confidence for no flip
    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)
    
    anchors = compute_anchors(obj['new_alpha'])
    
    for anchor in anchors:
        orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1.
        
    confidence = confidence / np.sum(confidence)
        
    obj['orient'] = orientation
    obj['conf'] = confidence
        
    # Fix orientation and confidence for flip
    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)
    
    anchors = compute_anchors(2.*np.pi - obj['new_alpha'])
    
    for anchor in anchors:
        orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1
        
    confidence = confidence / np.sum(confidence)
        
    obj['orient_flipped'] = orientation
    obj['conf_flipped'] = confidence

def prepare_input_and_output(train_inst):
    ### Prepare image patch
    xmin = train_inst['xmin']
    ymin = train_inst['ymin']
    xmax = train_inst['xmax']
    ymax = train_inst['ymax']
    
    img = cv2.imread(image_dir + train_inst['image'])
    img = copy.deepcopy(img[ymin:ymax+1, xmin:xmax+1]).astype(np.float32)
    
    # Resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img - np.array([[[103.939, 116.779, 123.68]]])
    
    ### Fix orientation and confidence
    flip = np.random.binomial(1, .5)
    if flip > 0.5:
        img = cv2.flip(img, 1)
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf']

def augment_image(image):
    augmented_image = seq.augment_images([image])[0]
    return augmented_image

def data_gen(all_objs, batch_size):
    num_obj = len(all_objs)
    keys = list(range(num_obj))
    np.random.shuffle(keys)

    l_bound = 0
    r_bound = batch_size if batch_size < num_obj else num_obj

    while True:
        if l_bound == r_bound:
            l_bound = 0
            r_bound = batch_size if batch_size < num_obj else num_obj
            np.random.shuffle(keys)

        if not AUGMENTATION:
            x_batch = np.zeros((batch_size, 224, 224, 3))
            d_batch = np.zeros((batch_size, 3))
            o_batch = np.zeros((batch_size, BIN, 2))
            c_batch = np.zeros((batch_size, BIN))

            for idx, key in enumerate(keys[l_bound:r_bound]):
                # Input image and fix object's orientation and confidence
                image, dimension, orientation, confidence = prepare_input_and_output(all_objs[key])

                # Original images
                x_batch[idx, :] = image
                d_batch[idx, :] = dimension
                o_batch[idx, :] = orientation
                c_batch[idx, :] = confidence

            yield x_batch, [d_batch, o_batch, c_batch]

        if AUGMENTATION:
            x_batch = np.zeros((2 * batch_size, 224, 224, 3))
            d_batch = np.zeros((2 * batch_size, 3))
            o_batch = np.zeros((2 * batch_size, BIN, 2))
            c_batch = np.zeros((2 * batch_size, BIN))

            for idx, key in enumerate(keys[l_bound:r_bound]):
                # Input image and fix object's orientation and confidence
                image, dimension, orientation, confidence = prepare_input_and_output(all_objs[key])

                # Original images
                x_batch[idx, :] = image
                d_batch[idx, :] = dimension
                o_batch[idx, :] = orientation
                c_batch[idx, :] = confidence

                # Augmented images
                x_batch[idx + batch_size, :] = augment_image(image)
                d_batch[idx + batch_size, :] = dimension.copy()
                o_batch[idx + batch_size, :] = orientation.copy()
                c_batch[idx + batch_size, :] = confidence.copy()

            yield x_batch, [d_batch, o_batch, c_batch]

        l_bound = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_obj:
            r_bound = num_obj

def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)

######## Regression Network #######
input_image = Input(shape=(NORM_H, NORM_W, 3))

if select_model == 'resnet18':
    backbone = ResNet18(input_tensor=input_image, include_top=False)
    x = backbone.output
elif select_model == 'resnet50':
    backbone = ResNet50(input_tensor=input_image, include_top=False)
    x = backbone.output
elif select_model == 'resnet101':
    backbone = ResNet101(input_tensor=input_image, include_top=False)
    x = backbone.output
elif select_model == 'resnet152':
    backbone = ResNet152(input_tensor=input_image, include_top=False)
    x = backbone.output
elif select_model == 'vgg11':
    backbone = VGG11(input_tensor=input_image, include_top=False)
    x = backbone.output
elif select_model == 'vgg16':
    backbone = VGG16(input_tensor=input_image, include_top=False)
    x = backbone.output
elif select_model == 'vgg19':
    backbone = VGG19(input_tensor=input_image, include_top=False)
    x = backbone.output
elif select_model == 'efficientnetb0':
    backbone = EfficientNetB0(input_tensor=input_image, include_top=False)
    x = backbone.output
elif select_model == 'efficientnetb5':
    backbone = EfficientNetB5(input_tensor=input_image, include_top=False)
    x = backbone.output
elif select_model == 'mobilenetv2':
    backbone = MobileNetV2(input_tensor=input_image, include_top=False)
    x = backbone.output
else:
    raise ValueError(f"Invalid model selection: {select_model}")

x = GlobalAveragePooling2D()(x)

dimension_output = Dense(3, name='dimension')(x)
orientation_output = Dense(BIN * 2)(x)
orientation_output = Reshape((BIN, 2))(orientation_output)
orientation_output = Lambda(l2_normalize, name='orientation')(orientation_output)
confidence_output = Dense(BIN, activation='softmax', name='confidence')(x)

model = Model(inputs=input_image, outputs=[dimension_output, orientation_output, confidence_output])

model.summary()

##### Training Setup ######
alpha = 0.6
lr = 1e-4

def orient_loss(y_true, y_pred):
    return W * K.sum(y_true * y_pred) / BATCH_SIZE

def confidence_loss(y_true, y_pred):
    return ALPHA * K.categorical_crossentropy(y_true, y_pred)

def dimension_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

losses = {
    'dimension': dimension_loss,
    'orientation': orient_loss,
    'confidence': confidence_loss,
}

loss_weights = {'dimension': 0.4, 'orientation': 0.3, 'confidence': 0.3}
    
model.compile(optimizer=Adam(learning_rate=lr), loss=losses, loss_weights=loss_weights)

###### Callbacks ######
# Define where the model should be saved
checkpoint_path = "./checkpoints/resnet50_weights.h5"

# Create a ModelCheckpoint callback to save the model
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

# You can also define an EarlyStopping callback to stop training if no improvement is seen
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='min')

# Use these callbacks in your model's `fit` method
train_gen = data_gen(all_objs, BATCH_SIZE)

model.fit(train_gen, steps_per_epoch=100, epochs=10, callbacks=[checkpoint, early_stopping])
