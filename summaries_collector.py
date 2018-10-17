import os
import tensorflow as tf


class SummariesCollector:
    def __init__(self, summaries_dir, model_name):
        self._train_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train_' + model_name))
        self.write_train_episode_summaries = self._init_episode_summaries('train', self._train_summary_writer)
        self.write_train_curriculum_summaries = self._init_curriculum_summaries('train', self._train_summary_writer)

        self._test_summary_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'test_' + model_name))
        self.write_test_episode_summaries = self._init_episode_summaries('test', self._test_summary_writer)
        self.write_test_curriculum_summaries = self._init_curriculum_summaries('test', self._test_summary_writer)

    @staticmethod
    def _init_episode_summaries(prefix, summary_writer):
        episodes_played_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        successful_episodes_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        collision_episodes_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        max_length_episodes_var = tf.Variable(0, trainable=False, dtype=tf.float32)

        summaries = tf.summary.merge([
            tf.summary.scalar(prefix + '_episodes_played', episodes_played_var),
            tf.summary.scalar(prefix + '_successful_episodes', successful_episodes_var / episodes_played_var),
            tf.summary.scalar(prefix + '_collision_episodes', collision_episodes_var / episodes_played_var),
            tf.summary.scalar(prefix + '_max_len_episodes', max_length_episodes_var / episodes_played_var)
        ])

        def write_episode_summaries(sess, global_step, episodes_played, successful_episodes, collision_episodes,
                                    max_len_episodes):
            summary_str = sess.run(summaries, feed_dict={
                episodes_played_var: episodes_played,
                successful_episodes_var: successful_episodes,
                collision_episodes_var: collision_episodes,
                max_length_episodes_var: max_len_episodes
            })

            summary_writer.add_summary(summary_str, global_step)
            summary_writer.flush()

        return write_episode_summaries

    @staticmethod
    def _init_curriculum_summaries(prefix, summary_writer):
        curriculum_status_var = tf.Variable(0, trainable=False, dtype=tf.float32)
        summaries = tf.summary.scalar(prefix + '_curriculum_status', curriculum_status_var)

        def write_curriculum_summaries(sess, global_step, status):
            if status is None:
                return
            summary_str = sess.run(summaries, feed_dict={curriculum_status_var: status})

            summary_writer.add_summary(summary_str, global_step)
            summary_writer.flush()

        return write_curriculum_summaries

    def write_train_optimization_summaries(
            self, critic_summaries, actor_summaries, reward_summaries, global_step
    ):
        self._train_summary_writer.add_summary(critic_summaries, global_step)
        self._train_summary_writer.add_summary(actor_summaries, global_step)
        if reward_summaries is not None:
            self._train_summary_writer.add_summary(reward_summaries, global_step)
        self._train_summary_writer.flush()



