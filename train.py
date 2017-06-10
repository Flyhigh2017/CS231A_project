import tensorflow as tf
import datetime
import os
import argparse
import config as cfg
from yolo_net import YOLONet
from kitti import Kitti

class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data #kitti object
        self.image_size1 = cfg.IMAGE_SIZE1
        self.image_size2 = cfg.IMAGE_SIZE2
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        #self.initial_learning_rate = cfg.LEARNING_RATE
        self.learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, 'Save_weight')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')

        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

#self.learning_rate = tf.train.exponential_decay(
#         self.initial_learning_rate, self.global_step, self.decay_steps,
#           self.decay_rate, self.staircase, name='learning_rate')

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(
            self.net.total_loss, global_step=self.global_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def train(self):
        tf.reset_default_graph()
        images, labels, ob_mask = self.data.get_batch()
        print "finish loading dataset"
        for step in xrange(1, self.max_iter + 1):
            flag = self.data.finish
            if flag == True:
                break
            #images, labels, ob_mask = self.data.get_batch()
            feed_dict = {self.net.images: images, self.net.labels: labels, self.net.ob_mask: ob_mask}
            loss, _ = self.sess.run([self.net.total_loss, self.optimizer],feed_dict=feed_dict)
            print "loss is", loss
            print "step is", step
            if step % 7 == 0:
                self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)
                print ("save weights", step)


def main():


    yolo = YOLONet()
    k = Kitti()
    #update_config_paths()
    solver = Solver(yolo, k)

    print('Start training ...')
    solver.train()
    print('Done training.')

if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
