import tensorflow as tf

flags = tf.app.flags

####################################
#    Hyper parameters              #
####################################

# for training
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epoch', 14, 'epoch')

############################
#   environment setting    #
############################
flags.DEFINE_string('data_dir', '../affMNIST_data', 'Dir checkpoints are saved')
flags.DEFINE_string('checkpoint_dir', 'checkpoint_dir', 'Dir checkpoints are saved')
flags.DEFINE_integer('save_summaries_steps', 500, 'Number steps elapsed for summary save')

flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_string('results', 'results', 'path for saving results')
flags.DEFINE_integer('train_sum_freq', 50, 'the frequency of saving train summary(step)')
#flags.DEFINE_integer('train_sum_freq', 50, 'the frequency of saving train summary(step)')
#flags.DEFINE_integer('train_sum_freq', 20, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 50, 'the frequency of saving valuation summary(step)')
#flags.DEFINE_integer('val_sum_freq', 1500, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_checkpoint_steps', 1000, 'save checkpoint every #(steps)')

############################
#   distributed setting    #
############################

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
