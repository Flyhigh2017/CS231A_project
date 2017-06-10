import numpy as np
import tensorflow as tf
import config as cfg




class YOLONet(object):
    
    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(cfg.CLASSES_LIST)
        self.image_width = cfg.IMAGE_SIZE2 #448
        self.image_height = cfg.IMAGE_SIZE1 #448
        self.cell_size1 = cfg.CELL_SIZE1 #5
        self.cell_size2 = cfg.CELL_SIZE2 #16
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size1 * self.cell_size2) * (self.num_class + self.boxes_per_cell * 5)
        self.scale1 = 1.0 * self.image_height / self.cell_size1
        self.scale2 = 1.0 * self.image_width / self.cell_size2
        #self.boundary1 = self.cell_size * self.cell_size * self.num_class
        #self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell
        
        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE
        
        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA
        self.testinter = None
        self.images = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, 3], name='images')
        self.predicts = self.build_network(self.images, alpha=self.alpha, is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size1, self.cell_size2, 5 * self.boxes_per_cell + self.num_class])
            self.ob_mask = tf.placeholder(tf.float32, [None, self.cell_size1, self.cell_size2, self.boxes_per_cell])
            self.total_loss= self.loss_layer(self.predicts, self.labels, self.ob_mask)

    def build_network(self,
                      images,
                      #num_outputs,
                      alpha,
                      #keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            conv0 = tf.layers.conv2d(inputs=images, filters=32, kernel_size=3, padding='SAME', activation=leaky_relu)
            maxpool1 = tf.layers.max_pooling2d(inputs=conv0, pool_size=2, strides=2) # 512,160,32
            
            conv2 = tf.layers.conv2d(inputs=maxpool1, filters=64, kernel_size=3, padding='SAME', activation=leaky_relu)
            maxpool3 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2) # 256,80,64
            
            conv4 = tf.layers.conv2d(inputs=maxpool3, filters=128, kernel_size=3, padding='SAME', activation=leaky_relu)
            conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=1, padding='SAME', activation=leaky_relu)
            conv6 = tf.layers.conv2d(inputs=conv5, filters=128, kernel_size=3, padding='SAME', activation=leaky_relu)
            maxpool7 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=2) # 128,40,128
            
            conv8 = tf.layers.conv2d(inputs=maxpool7, filters=256, kernel_size=3, padding='SAME', activation=leaky_relu)
            conv9 = tf.layers.conv2d(inputs=conv8, filters=128, kernel_size=1, padding='SAME', activation=leaky_relu)
            conv10 = tf.layers.conv2d(inputs=conv9, filters=256, kernel_size=3, padding='SAME', activation=leaky_relu)
            maxpool11 = tf.layers.max_pooling2d(inputs=conv8, pool_size=2, strides=2) # 64,20,256
            
            conv12 = tf.layers.conv2d(inputs=maxpool11, filters=512, kernel_size=3, padding='SAME', activation=leaky_relu)
            conv13 = tf.layers.conv2d(inputs=conv12, filters=256, kernel_size=1, padding='SAME', activation=leaky_relu)
            conv14 = tf.layers.conv2d(inputs=conv13, filters=512, kernel_size=3, padding='SAME', activation=leaky_relu)
            conv15 = tf.layers.conv2d(inputs=conv14, filters=256, kernel_size=1, padding='SAME', activation=leaky_relu)
            conv16 = tf.layers.conv2d(inputs=conv15, filters=512, kernel_size=3, padding='SAME', activation=leaky_relu)
            maxpool17 = tf.layers.max_pooling2d(inputs=conv12, pool_size=2, strides=2) # 32,10,512
            
            conv18 = tf.layers.conv2d(inputs=maxpool17, filters=1024, kernel_size=3, padding='SAME', activation=leaky_relu)
            conv19 = tf.layers.conv2d(inputs=conv18, filters=512, kernel_size=1, padding='SAME', activation=leaky_relu)
            #conv20 = tf.layers.conv2d(inputs=conv19, filters=1024, kernel_size=3, padding='SAME', activation=leaky_relu)
            conv21 = tf.layers.conv2d(inputs=conv19, filters=128, kernel_size=1, padding='SAME', activation=leaky_relu)
            #conv22 = tf.layers.conv2d(inputs=conv21, filters=1024, kernel_size=3, padding='SAME', activation=leaky_relu)
            #conv23 = tf.layers.conv2d(inputs=conv22, filters=1024, kernel_size=3, padding='SAME', activation=leaky_relu)
            #conv24 = tf.layers.conv2d(inputs=conv23, filters=1024, kernel_size=3, padding='SAME', activation=leaky_relu)
            #conv25 = tf.layers.conv2d(inputs=conv24, filters=1024, kernel_size=3, padding='SAME', activation=leaky_relu)
            #batch7 = tf.layers.batch_normalization(conv25, training=is_training)
            maxpool26 = tf.layers.max_pooling2d(inputs=conv21, pool_size=2, strides=2) # 16,5,1024
            #maxpool26 = tf.layers.max_pooling2d(inputs=maxpool26, pool_size=2, strides=2)
            predicts = tf.layers.conv2d(inputs=maxpool26, filters=8, kernel_size=1, padding='SAME', activation=leaky_relu)
            #predicts = tf.nn.relu(predicts)
            self.net2 = predicts
        return predicts

    
    def loss_layer(self, predicts, ground_truth, mask, scope='loss_layer'):
        # predicts size batch x 6 x 6 x (5 + 5 + 3), 13 is filter depth
        # mask: batch x 6 x 6 x 1
        # ground_truth = batch x 6 x 6 x ()
        lam_coord=self.coord_scale
        lam_noobj=self.noobject_scale
        print predicts.get_shape(), ground_truth.get_shape(), mask.get_shape()
        with tf.variable_scope(scope):
            #x1h,y1h,w1h,h1h,cf1h, x2h,y2h,w2h,h2h,cf2h = ground_truth[:,:,:,:10]
            x1h = ground_truth[:,:,:,0] / 64.0
            y1h = ground_truth[:,:,:,1] / 64.0
            w1h = ground_truth[:,:,:,2] / self.image_width
            h1h = ground_truth[:,:,:,3] / self.image_height
            real_conf = ground_truth[:,:,:,4]
            '''
            x2h = ground_truth[:,:,:,5] / 64.0#self.image_width
            y2h = ground_truth[:,:,:,6] / 64.0#self.image_height
            w2h = ground_truth[:,:,:,7] / 64.0#self.image_width
            h2h = ground_truth[:,:,:,8] / 64.0#self.image_height
            cf2h = ground_truth[:,:,:,9]
            '''

            x1= predicts[:,:,:,0] #/ self.image_width
            y1= predicts[:,:,:,1] #/ self.image_height
            w1 = predicts[:,:,:,2] #/ self.image_width
            h1 = predicts[:,:,:,3] #/ self.image_height
            predicts_conf = predicts[:,:,:,4]
            '''
            x2 = predicts[:,:,:,5] #/ self.image_width
            y2 = predicts[:,:,:,6] #/ self.image_height
            w2 = predicts[:,:,:,7] #/ self.image_width
            h2 = predicts[:,:,:,8] #/ self.image_height
            cf2 = predicts[:,:,:,9]
            '''

            Ch = ground_truth[:,:,:,5:]
            C = predicts[:,:,:,5:]
            indicator1 = mask[:,:,:,0]
            #indicator2 = mask[:,:,:,1]
            print x1.get_shape()
            l11 = tf.multiply(indicator1, tf.square(x1-x1h)+tf.square(y1-y1h))
            #l12 = tf.multiply(indicator2, tf.square(x2-x2h)+tf.square(y2-y2h))
            coord_loss = lam_coord*tf.reduce_sum(l11)
            
            w1root,h1root = w1,h1#,w2,h2#tf.sqrt(w1),tf.sqrt(h1), tf.sqrt(w2),tf.sqrt(h2)
            w1hroot, h1hroot = w1h,h1h#,w2h,h2h#tf.sqrt(w1h), tf.sqrt(h1h), tf.sqrt(w2h), tf.sqrt(h2h)
            l21 = tf.multiply(indicator1, tf.square(w1root-w1hroot)+tf.square(h1root-h1hroot))
            #l22 = tf.multiply(indicator2, tf.square(w2root-w2hroot)+tf.square(h2root-h2hroot))
            coord_loss += 2*lam_coord*tf.reduce_sum(l21)
            ###
            #tf.losses.add_loss(coord_loss)
            
            l3_4 = tf.square(C-Ch)
            print indicator1.get_shape(), l3_4.get_shape()
            c1 = l3_4[:,:,:,0]
            c2 = l3_4[:,:,:,1]
            c3 = l3_4[:,:,:,2]
            class_loss = 0.4 * tf.reduce_sum(tf.multiply(indicator1, c1) + tf.multiply(indicator1, c2) + tf.multiply(indicator1, c3))
            #object_loss = tf.reduce_sum(tf.multiply(indicator1, tf.square(real_conf-predicts_conf)))
            #noobject_loss = lam_noobj*tf.reduce_sum(tf.multiply(1-indicator1, tf.square(real_conf - predicts_conf)))# +
            ###
            '''
            #tf.losses.add_loss(object_loss)
            x1h = label_relative_img[:,:,:,0]
            y1h = label_relative_img[:,:,:,1]
            w1h = label_relative_img[:,:,:,2]
            h1h = label_relative_img[:,:,:,3]
            pre_xleft, pre_ytop, real_xleft, real_ytop = x1-w1/2.0, y1-h1/2.0, x1h-w1h/2.0, y1h-h1h/2.0 # upper left(0,0)
            #rux, ruy, ruxh, ruyh = x1+w1/2, y1-h1/2, x1h+w1h/2, y1h-h1h/2 # upper right
            pre_xright, pre_ybottom, real_xright, real_ybottom = x1+w1/2.0, y1+h1/2.0, x1h+w1h/2.0, y1h+h1h/2.0 # bottom right
            #lbx, lby, lbxh, lbyh = x1+w1/2, y1+h1/2, x1h+w1h/2, y1h+h1h/2 # bottom left
            inter_xleft, inter_ytop = tf.maximum(real_xleft,pre_xleft), tf.maximum(pre_ytop,real_ytop) # upper left corner
            #rucx, rucy = tf.minimum(rux,ruxh), tf.maximum(ruy,ruyh) # upper right corner
            inter_xright, inter_ybottom = tf.minimum(real_xright,pre_xright), tf.minimum(real_ybottom,pre_ybottom) # bottom right corner
            #lbcx, lbcy = tf.maximum(lbx,lbxh), tf.minimum(lby,lbyh) # bottom left corner
            diffw1, diffh1 = tf.abs(inter_xright-inter_xleft), tf.abs(inter_ybottom-inter_ytop)
            diffw1 = tf.maximum(diffw1,0.0)
            diffh1 = tf.maximum(diffh1,0.0)
            intersection1 = tf.to_float(diffw1 * diffh1)
            self.testinter = intersection1
            #intersection1 += (1e-7)*tf.ones_like(intersection1)
            area1, area1h = tf.multiply(w1,h1), tf.multiply(w1h,h1h)
            union_square = area1 + area1h - intersection1
            iou_predict_truth = 1.0 * intersection1 / union_square
            self.testinter = union_square
            object_delta = indicator1 * (predicts_conf - iou_predict_truth)
            object_loss = tf.reduce_sum(tf.square(object_delta))
            noobject_delta = (1 - indicator1) * (predicts_conf - iou_predict_truth)
            noobject_loss = lam_noobj*tf.reduce_sum(tf.square(noobject_delta))
            '''
#       tf.losses.add_loss(class_loss)
            return coord_loss + class_loss


def leaky_relu(inputs, alpha=0.1):
    return tf.maximum(alpha*inputs, inputs)
