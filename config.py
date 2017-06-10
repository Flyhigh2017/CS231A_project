import os

#
# path and dataset parameter
#

DATA_PATH = '/Users/anekisei/Documents/cs231a_project/data/data_tracking_image_2/training/image_02'

LABEL_PATH = '/Users/anekisei/Documents/cs231a_project/data/label_02'


OUTPUT_DIR = os.path.join(DATA_PATH, 'output1')

WEIGHTS_DIR = os.path.join(DATA_PATH, 'weights1')

WEIGHTS_FILE = None
#WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights1', 'YOLO_small.ckpt')

CLASSES = {'Car': '0', 'Truck' : '1', 'Pedestrian' : '2'}
CLASSES_LIST = ['Car', 'Truck', 'Pedestrian']

FLIPPED = True


#
# model parameter
#
ORI_HEIGHT = 1200

ORI_WIDTH = 1920

IMAGE_SIZE1 = 448

IMAGE_SIZE2 = 448

CELL_SIZE = 7

CELL_SIZE1 = 7

CELL_SIZE2 = 7

BOXES_PER_CELL = 1

ALPHA = 0.1

DISP_CONSOLE = True

OBJECT_SCALE = 5.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 10

MAX_ITER = 140

SUMMARY_ITER = 1

SAVE_ITER = 1


#
# test parameter
#

THRESHOLD = 0.0

IOU_THRESHOLD = 0.0
