import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import cfg
from load_data import load_centered
from leNet import LeNet
#from tensorflow.python import debug as tf_debug

# supress tensorflow warning
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return(fd_train_acc, fd_loss, fd_val_acc)
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return(fd_test_acc)


def train(model, session):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_centered(is_training=True)

    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with session as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print('Training for epoch ' + str(epoch) + '/' + str(cfg.epoch) + ':')
            if session.should_stop():
                print('session stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc = sess.run(
                            [model.train_op, model.loss, model.accuracy])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(
                                model.accuracy,
                                { model.images: valX[start:end],
                                  model.tmp_labels: valY[start:end] })
                        val_acc += acc
                    val_acc = val_acc / num_val_batch
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()


def evaluation(model, session):
    teX, teY, num_te_batch = load_centered(cfg.dataset, cfg.batch_size, is_training=False)
    fd_test_acc = save_to()
    with session.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        session.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size

            acc, _ = sess.run([model.accuracy, model.train_summary],
                              {model.X: teX[start:end], model.labels: teY[start:end]})

            test_acc += acc

        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_accuracy.txt')


def main(_):
    graph = tf.Graph()
    with graph.as_default():
        model = LeNet()
        
        if cfg.is_training:
            session = tf.train.MonitoredTrainingSession(
                          checkpoint_dir=cfg.checkpoint_dir,
                          save_checkpoint_secs=cfg.save_checkpoint_secs,
                          save_summaries_steps=cfg.save_summaries_steps,
                          hooks=[tf.train.NanTensorHook(model.loss),
                                 tf.train.StepCounterHook(),
                                 tf.train.SummarySaverHook(save_steps=cfg.train_sum_freq,
                                                           output_dir=cfg.logdir,
                                                           summary_op=model.train_summary)],
                      )
            train(model, session)
        else:
            session = tf.train.MonitoredTrainingSession(
                          checkpoint_dir=cfg.checkpoint_dir,
                          save_checkpoint_secs=cfg.save_checkpoint_secs,
                          save_summaries_steps=cfg.save_summaries_steps,
                      )
            evaluation(model, session)

if __name__ == "__main__":
    tf.app.run()
