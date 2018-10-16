import tensorflow as tf
import os
import datetime
import time


class SaverWrapper:
    def __init__(self, saver_dir):
        self.saver_dir = saver_dir
        if not os.path.exists(self.saver_dir):
            os.makedirs(self.saver_dir)
        now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
        self.saver_path = os.path.join(self.saver_dir, now)
        self.saver = tf.train.Saver(max_to_keep=3, save_relative_paths=self.saver_dir)
        print('models will be saved to: {}'.format(self.saver_dir))

    def load_model(self, sess):
        checkpoint_path = tf.train.get_checkpoint_state(self.saver_dir)
        if checkpoint_path is not None:
            self.saver.restore(sess, checkpoint_path.model_checkpoint_path)
            print('Model restored from file: {}'.format(checkpoint_path.model_checkpoint_path))
        else:
            print('Model not found in: {}'.format(self.saver_dir))

    def save_model(self, sess, save_retries=3, global_step=None):
        for i in range(save_retries):
            try:
                # save model
                self.saver.save(sess, self.saver_path, global_step=global_step)
                print('Model saved')
                return True
            except:
                pass
        print('Failed to save model')
        return False
