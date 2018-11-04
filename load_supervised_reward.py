import yaml

from pre_trained_reward import *

if __name__ == '__main__':
    # model_name = '2018_10_26_15_00_56'
    # model_name = '2018_10_29_16_14_37'
    model_name = '2018_10_31_10_44_31'

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
