import keras
import numpy as np
from numpy import asarray
import pandas as pd
import json
import cv2
import os
from PIL import Image
from keras.applications.mobilenet import preprocess_input
from keras.metrics import top_k_categorical_accuracy

size = 64
batchsize = 680
BASE_SIZE = 256

def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img


def list_all_categories():
    return ['airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan', 'cell phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup', 'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee', 'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog', 'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint can', 'paintbrush', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote control', 'rhinoceros', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

# def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
#     while True:
#         for k in np.random.permutation(ks):
#             filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
#             for df in pd.read_csv(filename, chunksize=batchsize):
#                 df['drawing'] = df['drawing'].apply(json.loads)
#                 x = np.zeros((len(df), size, size, 1))
#                 for i, raw_strokes in enumerate(df.drawing.values):
#                     x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
#                                              time_color=time_color)
#                 x = preprocess_input(x).astype(np.float32)
#                 y = keras.utils.to_categorical(df.y, num_classes=NCATS)
#                 yield x, y

def df_to_image_array_xd(df, size, lw=6, time_color=True):
    ytest = draw_cv2(df.drawing.values[0], size=size, lw=lw, time_color=False)
    np.savetxt("variable.txt", ytest)
    

    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    cv2.imwrite('opncv_kolala.png', x[0])
    x = preprocess_input(x).astype(np.float32)
    return x

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

model = keras.models.load_model("modelNewV.h5",custom_objects={"top_3_accuracy": top_3_accuracy})

test = pd.read_csv('test_simplified.csv')
test1= test.head(1)
x_test = df_to_image_array_xd(test1, size)
# print(x_test.shape)
# Reshape the 4D array to 2D
# flattened_array = x_test.reshape((-1, x_test.shape[-1]))
# # Save the flattened array to a file in text format (.txt or .csv)
# np.savetxt("variable.txt", x_test)

im = cv2.imread('cut.png', 0)
# im = cv2.resize(im, (64, 64))

x = np.zeros((1, size, size, 1))
x[0, :, :, 0] = im
cv2.imwrite('firstTry.png', x[0])
xImage = preprocess_input(x).astype(np.float32)

test_predictions = model(xImage)
# test_predictions = model.predict(x_test, batch_size=128, verbose=1)
top3 = preds2catids(test_predictions)
print(type(test_predictions))
# print(top3)
cats = list_all_categories()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
top3cats = top3.replace(id2cat)
# print(top3cats)
# print(top3cats.shape)


# testData = [[[17, 18, 20, 25, 137, 174, 242, 249, 251, 255, 251, 229, 193, 166, 104, 58, 25, 13, 3], [117, 176, 184, 185, 185, 190, 191, 187, 179, 122, 114, 103, 103, 109, 109, 100, 98, 103, 112]], [[64, 39, 25, 24, 37, 73, 78, 88, 91, 91, 84], [117, 117, 134, 155, 177, 180, 176, 160, 148, 129, 127]], [[203, 188, 181, 175, 174, 188, 207, 219, 225, 226, 215], [122, 120, 127, 137, 160, 169, 173, 161, 145, 133, 128]], [[110, 111, 151, 154, 154, 143, 108], [133, 150, 151, 150, 130, 127, 128]], [[0, 7, 18, 20, 28], [0, 10, 59, 80, 100]]]
# X = np.asarray(testData).astype(np.float32)
# Print the model summary
# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]

