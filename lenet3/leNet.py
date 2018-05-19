import tensorflow as tf

from config import cfg
from load_data import get_batch

epsilon = 1e-9


class LeNet(object):
    '''Modified LeNet-5'''

    def __init__(self, use_just_centered=True, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.images, self.tmp_labels = get_batch(use_just_centered, is_training)
                self.labels = tf.one_hot(self.tmp_labels, depth=10, axis=1, dtype=tf.uint8)

                self.build_arch()
                self.loss()
                self._accuracy()
                self._summary()

                self.global_step = tf.train.get_or_create_global_step()
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            else:
                self.images = tf.placeholder(tf.uint8, shape=(cfg.batch_size, 40, 40, 1))
                self.labels = tf.placeholder(tf.uint8, shape=(cfg.batch_size, 10, 1))
                self.build_arch()


    def build_arch(self):
        with tf.variable_scope('c1'):
            c1 = tf.layers.conv2d(self.images,
                                  filters=6,
                                  kernel_size=5,
                                  strides=1)
            assert c1.shape == (cfg.batch_size, 36, 36, 6)

        with tf.variable_scope('s2'):
            # TODO Use sub-sampling as defined on paper, not max-pooling
            s2 = tf.layers.max_pooling2d(c1,
                                         pool_size=2,
                                         strides=2)
            assert s2.shape == (cfg.batch_size, 18, 18, 6)

        with tf.variable_scope('c3'):
            c3 = tf.layers.conv2d(s2,
                                  filters=16,
                                  kernel_size=5,
                                  strides=1)
            assert c3.shape == (cfg.batch_size, 14, 14, 16)

        with tf.variable_scope('s4'):
            # TODO Use sub-sampling as defined on paper, not max-pooling
            s4 = tf.layers.max_pooling2d(c3,
                                         pool_size=2,
                                         strides=2)
            assert s4.shape == (cfg.batch_size, 7, 7, 16)

        with tf.variable_scope('flat5'):
            flat5 = tf.reshape(s4, [-1, 7 * 7 * 16])
            assert flat5.shape == (cfg.batch_size, 7 * 7 * 16)

        with tf.variable_scope('f6'):
            f6 = tf.contrib.layers.fully_connected(flat5, num_outputs=84)
            assert f6.shape == (cfg.batch_size, 84)

        with tf.variable_scope('output'):
            self.output = tf.contrib.layers.fully_connected(f6, num_outputs=10)
            assert self.output.shape == (cfg.batch_size, 10)


    def loss(self):
        diff = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.output)
        self.loss = tf.reduce_mean(diff)

    
    def _accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def _summary(self):
        tf.summary.image('images', self.images)
        tf.summary.scalar('train/loss', self.loss)
        tf.summary.scalar('train/accuracy', self.accuracy)
        self.train_summary = tf.summary.merge_all()
