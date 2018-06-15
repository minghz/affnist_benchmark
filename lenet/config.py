import tensorflow as tf

flags = tf.app.flags

####################################
#    Hyper parameters              #
####################################

# for training
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epoch', 50, 'epoch')

############################
#   environment setting    #
############################
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')

flags.DEFINE_string('checkpoint_dir', 'checkpoint_dir', 'Dir checkpoints are saved')
flags.DEFINE_integer('save_checkpoint_steps', 1000, 'save checkpoint every #(steps)')

flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_string('peppered', '0', 'affmnist peppered with transformed images of such percentage')
flags.DEFINE_string('centered', '2', 'affmnist centered images, percent of 60k')
flags.DEFINE_string('affmnist_data_dir', '../affMNIST_data', 'Dir for affmnist data')

flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 5, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 5, 'the frequency of saving valuation summary(step)')
flags.DEFINE_string('results', 'results', 'path for saving results')

############################
#   distributed setting    #
############################

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
