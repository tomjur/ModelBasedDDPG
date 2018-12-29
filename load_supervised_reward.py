import yaml
from openrave_manager import OpenraveManager
from potential_point import PotentialPoint

from pre_trained_reward import *


def print_model_stats(pre_trained_reward_network, test_batch_size, sess):
    # read the data
    test = load_data_from(os.path.join('supervised_data', 'test'), max_read=10 * test_batch_size)
    print len(test)

    # partition to train and test
    random.shuffle(test)

    openrave_manager = OpenraveManager(0.001, PotentialPoint.from_config(pre_trained_reward_network.config))

    sess.run(tf.global_variables_initializer())

    # run test for one (random) batch
    random.shuffle(test)
    test_batch = oversample_batch(test, 0, test_batch_size)
    test_batch, test_rewards, test_status = get_batch_and_labels(test_batch, openrave_manager)
    reward_prediction, status_prediction = pre_trained_reward_network.make_prediction(*([sess] + test_batch))
    # see what happens for different reward classes:
    goal_rewards_stats, collision_rewards_stats, other_rewards_stats = compute_stats_per_class(
        test_status, test_rewards, status_prediction, reward_prediction)
    print 'before loading weights'
    print 'goal mean_error {} max_error {} accuracy {}'.format(*goal_rewards_stats)
    print 'collision mean_error {} max_error {} accuracy {}'.format(*collision_rewards_stats)
    print 'other mean_error {} max_error {} accuracy {}'.format(*other_rewards_stats)

    # load weights
    pre_trained_reward_network.load_weights(sess)
    # run test for one (random) batch
    random.shuffle(test)

    test_batch = oversample_batch(test, 0, test_batch_size)
    test_batch, test_rewards, test_status = get_batch_and_labels(test_batch, openrave_manager)
    reward_prediction, status_prediction = pre_trained_reward_network.make_prediction(*([sess] + test_batch))
    # see what happens for different reward classes:
    goal_rewards_stats, collision_rewards_stats, other_rewards_stats = compute_stats_per_class(
        test_status, test_rewards, status_prediction, reward_prediction)
    print 'after loading weights'
    print 'goal mean_error {} max_error {} accuracy {}'.format(*goal_rewards_stats)
    print 'collision mean_error {} max_error {} accuracy {}'.format(*collision_rewards_stats)
    print 'other mean_error {} max_error {} accuracy {}'.format(*other_rewards_stats)


if __name__ == '__main__':
    # model_name = '2018_10_26_15_00_56'
    # model_name = '2018_10_29_16_14_37'
    # model_name = '2018_10_31_10_44_31' # for easy workspace
    # model_name = '2018_11_10_09_02_38' # for hard workspace
    model_name = '2018_12_05_13_46_51'  # hard workspace with varying actions

    # read the config
    config_path = os.path.join(os.getcwd(), 'config/reward_config.yml')
    with open(config_path, 'r') as yml_file:
        config = yaml.load(yml_file)
        print('------------ Config ------------')
        print(yaml.dump(config))

    test_batch_size = config['model']['batch_size'] * 10

    # create the network
    pre_trained_reward_network = PreTrainedReward(model_name, config)

    with tf.Session(
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config['general']['gpu_usage'])
            )
    ) as sess:
        print_model_stats(pre_trained_reward_network, test_batch_size, sess)
