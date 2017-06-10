import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import cPickle
import copy
import config as cfg
from os import listdir
from os.path import isfile, join
import tensorflow as tf

class Kitti(object):
    def __init__(self):
        self.data_path = cfg.DATA_PATH
        self.label_path = cfg.LABEL_PATH
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.image_path = join(self.data_path, 'object-detection-crowdai')
        self.image_width = cfg.IMAGE_SIZE1
        self.image_height = cfg.IMAGE_SIZE2
        self.file_names = [ f for f in listdir(self.image_path) if isfile(join(self.image_path,f)) ]
        self.widthRatio = (self.image_width + 0.0) / (cfg.ORI_WIDTH + 0.0)
        self.heightRatio = (self.image_height + 0.0) / (cfg.ORI_HEIGHT + 0.0)
        self.class_dic = cfg.CLASSES
        self.cell_size1 = cfg.CELL_SIZE1
        self.cell_size2 = cfg.CELL_SIZE2
        self.num_class = len(cfg.CLASSES_LIST)
        self.batch_start = 0
        self.batch_size = cfg.BATCH_SIZE
        self.finish = False
        #self.label_temp = self.label_load(self.label_path, self.class_dic, self.data)
        #self.label, self.mask = self.label_transfer_mask(self.data, self.label_temp, self.cell_size1, self.cell_size2, self.num_class)  #label : N x 5 x 16 x (5x2+3), object_mask : (N x 5 x 16 x 2)

    def genrate_matrix(self,im_list):
        N = len(im_list)
        h, w, channel = im_list[0].shape
        X_train = np.zeros(shape=(N,h,w,channel))
        for i in range(N):
            X_train[i,:,:,:] =im_list[i]
        return X_train


    def image_read(self, mypath, width, height, batch_start):#modify to list[matrix] in the future width=1024
        onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
        images = []
        start_index = batch_start * self.batch_size
        for n in range(start_index, start_index + self.batch_size):
            if (onlyfiles[n][-1] == 'g') and (n < len(onlyfiles)):
                raw_image = cv2.imread( join(mypath,onlyfiles[n]) )
                resized_image = cv2.resize(raw_image, (width, height))
                imag = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.float32)
                imag = (imag / 255.0) * 2.0 - 1.0
                images.append(imag)
    #dataset = self.genrate_matrix(images)
        return images

    def get_batch(self):
        images_list = self.image_read(self.image_path, self.image_width, self.image_height, self.batch_start)
        images_matrix = self.genrate_matrix(images_list)
        #label_batch_list = self.label_list[self.batch_start * self.batch_size : self.batch_start * self.batch_size + self.batch_size]
        label_batch_list = self.label_load_batch(self.label_path, self.class_dic, self.batch_start)
        labels, object_mask = self.label_transfer_mask(images_matrix, label_batch_list, self.cell_size1, self.cell_size2, self.num_class)
        self.batch_start += 1
        if self.batch_size * (self.batch_start + 1) > len(self.file_names):
            self.finish = True
        return images_matrix, labels, object_mask


    
    def label_load_batch(self, label_path, classes, batch_start):#modify to list[matrix] in the future
        label_matrix = np.genfromtxt(os.path.join(label_path, 'labels.csv'),delimiter=',',dtype='str')
        index_list = []
        label_matrix = np.delete(label_matrix,[0],0)
        batch_file_names = self.file_names[batch_start * self.batch_size : batch_start * self.batch_size + self.batch_size]
        print batch_file_names
        #xmin,xmax,ymin,ymax,Frame,Label,Preview URL
        frame_vec = label_matrix[:,4]
        #frame_vec = frame_vec.astype(np.float)
        leftTop_x = label_matrix[:,0]
        leftTop_y = label_matrix[:,1]
        rightBot_x = label_matrix[:,2]
        rightBot_y = label_matrix[:,3]
        
        
        leftTop_x = leftTop_x.astype(np.float)
        leftTop_y = leftTop_y.astype(np.float)
        rightBot_x = rightBot_x.astype(np.float)
        rightBot_y = rightBot_y.astype(np.float)
        
        leftTop_x = (leftTop_x * self.widthRatio).astype(np.int)
        leftTop_y = (leftTop_y * self.heightRatio).astype(np.int)
        rightBot_x = (rightBot_x * self.widthRatio).astype(np.int)
        rightBot_y = (rightBot_y * self.heightRatio).astype(np.int)
        
        center_x = ((leftTop_x + rightBot_x) / 2).astype(np.int)
        center_y = ((leftTop_y + rightBot_y) / 2).astype(np.int)
        width = np.absolute(rightBot_x - leftTop_x)
        height = np.absolute(rightBot_y - leftTop_y)
        
        type_vec = label_matrix[:,5]
        for i in range(type_vec.shape[0]):
            type_vec[i] = classes[type_vec[i]]
        
        type_vec = type_vec.astype(np.int)
        
        new_label = np.array([type_vec, leftTop_x, leftTop_y, rightBot_x, rightBot_y]).T
        center_label = np.array([type_vec, center_x, center_y, width, height]).T
        
        label_list_batch = []
        '''
        for i in range(N):
            img_name = self.file_names[i]
            singleImage_label_main = []
            while(frame_vec[start_index] == img_name):
                type, x_center, y_center, width, height = center_label[start_index,:]
                singleImage_label_main.append([type, x_center, y_center, width, height])
                if start_index == (frame_vec.shape[0] - 1) :
                    break
                start_index += 1
            singleImage_label_main = np.array(singleImage_label_main)
            singleImage_label_main = singleImage_label_main.reshape((-1,5))
            #print "haha:", singleImage_label_main
            label_list_main.append(singleImage_label_main)
        '''
        for i in range(len(batch_file_names)):
            img_name = batch_file_names[i]
            singleImage_label_main = []
            for j in range(label_matrix.shape[0]):
                if img_name == label_matrix[j,4]:
                    type, x_center, y_center, width, height = center_label[j,:]
                    singleImage_label_main.append([type, x_center, y_center, width, height])
            singleImage_label_main = np.array(singleImage_label_main)
            singleImage_label_main = singleImage_label_main.reshape((-1,5))
            label_list_batch.append(singleImage_label_main)

        return label_list_batch

    def label_transfer_mask(self, data, label_list_main, cell_size1, cell_size2, num_class): #cell_size1: height, cell_size2: width
        batch_size, H, W, C = data.shape
        labels = np.zeros(shape=(batch_size, cell_size1, cell_size2, 5 * self.boxes_per_cell + num_class))
        object_mask = np.zeros(shape=(batch_size, cell_size1, cell_size2, self.boxes_per_cell))
        cell_length = H / cell_size1
        for n in range(batch_size):
            singleImage_labels = label_list_main[n]
            for m in range(singleImage_labels.shape[0]):
                single_label = singleImage_labels[m,:]
                class_note = single_label[0]
                center_x = single_label[1]
                center_y = single_label[2]
                w = single_label[3] + 0.0
                h = single_label[4] + 0.0
                for i in range(cell_size1):
                    height_start = i * cell_length + 0.0
                    if (center_y >= height_start) and (center_y <= height_start + cell_length):
                        for j in range(cell_size2):
                            width_start = j * cell_length + 0.0
                            if (center_x >= width_start) and (center_x <= width_start + cell_length):
                                if labels[n,i,j,4] == 0:
                                    labels[n,i,j,0] = (center_x + 0.0 - width_start) #/ self.image_width
                                    labels[n,i,j,1] = (center_y + 0.0 - height_start) #/ self.image_height
                                    labels[n,i,j,2] = (w + 0.0) #/ self.image_width
                                    labels[n,i,j,3] = (h + 0.0) #/ self.image_height
                                    labels[n,i,j,4] = 1.0
                                    labels[n,i,j,5+class_note] = 1.0
                                    object_mask[n,i,j,0] = 1.0
                                
                                
                                '''
                                else:
                                    labels[n,i,j,5] = (center_x + 0.0 - width_start) #/ self.image_width
                                    labels[n,i,j,6] = (center_y + 0.0 - height_start) #/ self.image_height
                                    labels[n,i,j,7] = (w + 0.0) #/ self.image_width
                                    labels[n,i,j,8] = (h + 0.0) #/ self.image_height
                                    labels[n,i,j,9] = 1.0
                                    labels[n,i,j,10+class_note] = 1.0
                                    object_mask[n,i,j,1] = 1.0
                                '''
                    else:
                        continue

        return labels, object_mask#, label_relative_image #label : N x 5 x 16 x (5x2+3), object_mask : (N x 5 x 16 x 2)


#below has nothing to do with Kitti class
'''
    def calc_iou(self, boxes1, boxes2):
        """calculate ious
        Args:
        boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
        boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
        iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                           boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])
                               
        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                           boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])
                               
        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])
                               
        # intersection
        intersection = tf.maximum(tf.cast(0.0, tf.float64), tf.cast(rd - lu, tf.float64))
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
                               
        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
            (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
            (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])
                                       
        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
                                           
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


    def calculate_mask(self, predicts, labels): #predict: Nx5x16x13, label: Nx5x16x8
        batch_size, cell_size1, cell_size2, D = labels.shape
        batch_size, cell_size1, cell_size2, dim = predicts.shape
        predicts = tf.reshape(predicts, [batch_size, cell_size1 * cell_size2 * dim])
        predict_classes = tf.reshape(predicts[:, :self.boundary1], [batch_size, self.cell_size1, self.cell_size2, self.num_class])#3, 0 1 2
        predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2], [batch_size, self.cell_size1, self.cell_size2, self.boxes_per_cell])#2, 3 4
        predict_boxes = tf.reshape(predicts[:, self.boundary2:], [batch_size, self.cell_size1, self.cell_size2, self.boxes_per_cell, 4])#4, 5,6,7,8 / 9,10,11,12
        
        response = tf.reshape(labels[:, :, :, 4], [batch_size, self.cell_size1, self.cell_size2, 1])
        boxes = tf.reshape(labels[:, :, :, 0:4], [batch_size, self.cell_size1, self.cell_size2, 1, 4])
        boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / np.sqrt(self.image_width * self.image_height)
        classes = labels[:, :, :, 5:]
        predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] ) / self.cell_size2,#x
                                       (predict_boxes[:, :, :, :, 1] ) / self.cell_size1,#y
                                       tf.square(predict_boxes[:, :, :, :, 2]),#square(w)
                                       tf.square(predict_boxes[:, :, :, :, 3])])#square(h)
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])
                                           
        iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)
                                           
        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float64) * response
                                           
        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float64) - object_mask
        print object_mask.get_shape()
        return object_mask

'''













