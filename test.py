import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
import config as cfg
from os.path import isfile, join
import scipy.misc
from yolo_net import YOLONet


class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file
        self.data_path = cfg.DATA_PATH
        self.label_path = cfg.LABEL_PATH
        self.classes = cfg.CLASSES
        self.num_class = len(cfg.CLASSES_LIST)
        self.image_width = cfg.IMAGE_SIZE2
        self.image_height = cfg.IMAGE_SIZE1
        self.widthRatio = (self.image_width + 0.0) / (cfg.ORI_WIDTH + 0.0)
        self.heightRatio = (self.image_height + 0.0) / (cfg.ORI_HEIGHT + 0.0)
        self.cell_size1 = cfg.CELL_SIZE1
        self.cell_size2 = cfg.CELL_SIZE2
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print 'Restoring weights from: ' + self.weights_file
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        img = cv2.resize(img, (self.image_width, self.image_height))
        for i in range(len(result)):
            x = int(result[i,0]) # DOUBLE-CHECK THE DIMENSIONS
            y = int(result[i,1])
            w = int(result[i,2] / 2)
            h = int(result[i,3] / 2)
            type = int(result[i,4])
            if (y > self.image_height * 2.0 / 5.0) and y < (self.image_height * 2.0 / 3.0):
                cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img, (x - w, y - h - 20),(x + w, y - h), (125, 125, 125), -1)
                name = None
                if type == 0:
                    name = 'Car'
                if type == 1:
                    name = 'Truck'
                if type == 2:
                    name = 'Pedestrian'
                cv2.putText(img, name, (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.CV_AA)
        return img

    def detect(self, img):

        # format the image
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_width, self.image_height))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_height, self.image_width, 3)) # NOT SURE ABOUT WIDTH/HEIGHT
        result = self.detect_from_cvmat(inputs)[0]
        return result

    def detect_from_cvmat(self, inputs):
        # run the network
        net_output = self.sess.run(self.net.predicts,    #7x7x13
                                   feed_dict={self.net.images: inputs})
        test = net_output.reshape((120,8))
        results = []
        for i in range(net_output.shape[0]):
            #print "hello"
            shapes = net_output[i].shape #5 16 13
            results.append(self.interpret_output(net_output[i]))
        return results
    
    def interpret_output(self, output):
        probs = np.zeros((self.cell_size1, self.cell_size2,
                          self.boxes_per_cell, self.num_class))
        dim = output.shape
        trys = np.reshape(output,(dim[0]*dim[1],-1))
        
        scales = np.zeros(shape=(self.cell_size1, self.cell_size2, self.boxes_per_cell))
        scales[:,:,0] = output[:,:,4]
        scales = scales.reshape((120,1))
        class_probs = output[:,:,5:]
        softp = class_probs.reshape((120,3))

        sumd = np.sum(np.exp(softp),axis=1)
        sumd = sumd.reshape((120,1))
        #print sumd.shape
        softp = np.exp(softp)/sumd
        boxes = np.zeros(shape=(self.cell_size1, self.cell_size2, self.boxes_per_cell, 4))
        boxes[:,:,0,:] = output[:,:,0:4] #+ offset
        boxes = (boxes).astype(np.float32)
        dim = boxes.shape
        
        new_box = []
        scores = []
        class_predict = []
        count = 0
        for i in range(dim[0]):
            start_height = i * 128.0
            for j in range(dim[1]):
                box_class = softp[count,:]
                max_prob = np.amax(box_class)
                max_index = np.argmax(box_class)
                count += 1
                if max_prob > 0.57: #or p2[i,j] > 0.9 or p3[i,j] > 0.9:
                    start_width = j * 128.0
                    x_center = (boxes[i,j,0,0] * 128 + start_width)
                    y_center = (boxes[i,j,0,1] * 128 + start_height)
                    w = boxes[i,j,0,2] * self.image_width
                    h = boxes[i,j,0,3] * self.image_height
                    new_box.append([x_center, y_center, w, h])
                    scores.append(max_prob)
                    
                    max_index = np.argmax(box_class)
                    if max_index == 0:
                        class_predict.append(0.0)
                    if max_index == 1:
                        class_predict.append(1.0)
                    if max_index == 2:
                        class_predict.append(2.0)

        new_box = np.array(new_box)
        new_box = new_box.reshape((-1,4))
        #print new_box
        scores = np.array(scores)
        class_predict = np.array(class_predict)
        
        def non_max_suppression(bboxes, confidences, class_predict):
        # TODO: Implement this method!
            confidences.sort()
            confidences = confidences[::-1]
            class_predict = class_predict[::-1]
            bboxes = bboxes[::-1,:]
            bboxes = np.column_stack((bboxes,class_predict))
            nms_bboxes = []
            nms_bboxes.append(bboxes[0,:])
            for i in range(1,confidences.shape[0]):
                delete_flag = False
                x_center, y_center, width, height, type_veh = bboxes[i,:]
                #x_center = int((xmin + 0.0) + (width + 0.0) / 2)
                #y_center = int((ymin + 0.0) + (height + 0.0) / 2)
                for j in range(len(nms_bboxes)):
                    x_center1, y_center1, width1, height1, _ = nms_bboxes[j]
                    xmin1 = x_center1 - (width1 / 2.0)
                    ymin1 = y_center1 - (height1 / 2.0)
                    if (x_center <= (xmin1 + width1)) and (x_center >= xmin1):
                        if (y_center <= (ymin1 + height1)) and (y_center >= ymin1):
                            delete_flag = True
                            break
                if delete_flag == False:
                    nms_bboxes.append([x_center, y_center, width, height, type_veh])
        
            nms_bboxes = np.array(nms_bboxes)
            nms_bboxes = np.reshape(nms_bboxes,(-1,5))
            return nms_bboxes
        if new_box.shape[0] > 0:
            new_box = non_max_suppression(new_box,scores, class_predict)
        return new_box
    
    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)
    
                    
    
    def camera_detector(self, cap, wait=10):

        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            result = self.detect(frame)

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()
# detect ---> detect_from_cvmat ---> interpret_output
    def image_detector(self, imname, wait=0):
        image = cv2.imread(imname)
        result = self.detect(image)

        img = self.draw_result(image, result)
        cv2.imshow('Image', img)
        cv2.waitKey(wait)
        return result
        
    def label_load_image(self, label_path, file_name):#modify to list[matrix] in the future
        label_matrix = np.genfromtxt(os.path.join(label_path, 'labels.csv'),delimiter=',',dtype='str')
        label_matrix = np.delete(label_matrix,[0],0)
        #xmin,xmax,ymin,ymax,Frame,Label,Preview URL
        frame_vec = label_matrix[:,4]
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
        
        new_label = np.array([leftTop_x, leftTop_y, rightBot_x, rightBot_y]).T
        center_label = np.array([center_x, center_y, width, height]).T
        
        singleImage_label_main = []
        for j in range(label_matrix.shape[0]):
            if file_name == label_matrix[j,4]:
                leftTop_x, leftTop_y, rightBot_x, rightBot_y = new_label[j,:]
                singleImage_label_main.append([leftTop_x, leftTop_y, rightBot_x, rightBot_y])
        singleImage_label_main = np.array(singleImage_label_main)
        singleImage_label_main = singleImage_label_main.reshape((-1,4))

        return singleImage_label_main


def main():
    yolo = YOLONet(False)
    output_dir = os.path.join(cfg.OUTPUT_DIR, 'Save_weight')
    weight_file = os.path.join(output_dir, 'save.ckpt-140')
    detector = Detector(yolo, weight_file)
    test_data = ['1479498371963069978.jpg', '1479498372942264998.jpg', '1479498373462797835.jpg', '1479498373962951201.jpg', '1479498374962942172.jpg',
                '1479498375942206592.jpg', '1479498376463086347.jpg', '1479498377463264578.jpg', '1479498377963597629.jpg', '1479498378965237962.jpg']

    iou_total = 0.0
    iou_predict_truth = 0.0
    center_deviation = 0.0
    for m in range(len(test_data)):
        test_name = test_data[m]
        result = detector.image_detector(test_name)
        corres_label = detector.label_load_image(detector.label_path, test_name)
        for i in range(result.shape[0]):
            x_center, y_center, width, height, _ = result[i,:]
            for j in range(corres_label.shape[0]):
                leftTop_x, leftTop_y, rightBot_x, rightBot_y = corres_label[j,:]
                x_center_label = (leftTop_x + rightBot_x) / 2.0
                y_center_label = (leftTop_y + rightBot_y) / 2.0
                width_label = np.absolute(rightBot_x - leftTop_x)
                height_label = np.absolute(leftTop_y - rightBot_y)
                if np.sqrt((x_center - x_center_label)**2 + (y_center - y_center_label)**2) < 20:
                    center_deviation += np.sqrt((x_center - x_center_label)**2 + (y_center - y_center_label)**2)
                    box_predict = [x_center, y_center, width, height]
                    box_label = [x_center_label, y_center_label, width_label, height_label]
                    iou_predict_truth += detector.iou(box_predict, box_label)
                    iou_total += 1.0

    iou_accuracy = iou_predict_truth / iou_total
    average_center_deviation = center_deviation / iou_total
    print "iou accuracy is", iou_accuracy
    print "average center deviation is", average_center_deviation
    print "finish"


if __name__ == '__main__':
    main()
