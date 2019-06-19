import pickle
import os
import yaml
import datetime
import bz2
import numpy as np

from data_collector import CollectorProcess, DataCollector
from image_cache import ImageCache


class VisionRandomStepCollectorProcess(CollectorProcess):
    def _get_tuple(self, query_params=None):
        assert query_params is not None
        workspace_id = query_params[0]
        full_workspace_path = query_params[1]

        openrave_manager = self.openrave_interface.openrave_manager
        #  set the obstacles
        openrave_manager.set_params(full_workspace_path)

        # find free start and goal joints
        start_joints = openrave_manager.get_random_joints({0: 0.0})
        while not openrave_manager.is_valid(start_joints):
            start_joints = openrave_manager.get_random_joints({0: 0.0})

        goal_joints = openrave_manager.get_random_joints({0: 0.0})
        while not openrave_manager.is_valid(goal_joints):
            goal_joints = openrave_manager.get_random_joints({0: 0.0})

        # set fake trajectory with just start and goal, make sure the interface does not verify
        traj = [start_joints, goal_joints]
        self.openrave_interface.start_specific(traj, verify_traj=False)

        # take a random action
        random_action = np.random.uniform(-1.0, 1.0, len(start_joints) - 1)
        random_action /= np.linalg.norm(random_action)
        random_action = np.array([0.0] + list(random_action))
        next_joints, reward, terminated, status = self.openrave_interface.step(random_action)

        # the result contains also the workspace used
        return workspace_id, start_joints, goal_joints, random_action, next_joints, reward, terminated, status


class VisionRandomStepDataCollector(DataCollector):
    def _get_queue_size(self, number_of_threads):
        return 100*number_of_threads

    def _get_collector(self, config, queued_data_points, collector_specific_queue, params_file=None):
        return VisionRandomStepCollectorProcess(
            config, queued_data_points, self.results_queue, collector_specific_queue,
            query_parameters_queue=self.query_parameters_queue, init_rl_interface=True
        )


def print_status_dist(current_buffer):
    status = [t[-1] for t in current_buffer]
    total = len(status)
    for i in range(1, 4):
        count = sum([s == i for s in status])
        print '{}: {} ({})'.format(i, count, float(count) / total)


# read the config
config_path = os.path.join(os.getcwd(), 'config/config.yml')
with open(config_path, 'r') as yml_file:
    config = yaml.load(yml_file)
    print('------------ Config ------------')
    print(yaml.dump(config))

# scenario = 'vision'
scenario = 'vision_harder'
# worksapces_to_query = None
worksapces_to_query = ["922_workspace.pkl", "958_workspace.pkl", "765_workspace.pkl", "601_workspace.pkl", "608_workspace.pkl", "557_workspace.pkl", "770_workspace.pkl", "141_workspace.pkl", "937_workspace.pkl", "804_workspace.pkl", "309_workspace.pkl", "536_workspace.pkl", "768_workspace.pkl", "81_workspace.pkl", "256_workspace.pkl", "900_workspace.pkl", "50_workspace.pkl", "454_workspace.pkl", "513_workspace.pkl", "135_workspace.pkl", "368_workspace.pkl", "570_workspace.pkl", "112_workspace.pkl", "771_workspace.pkl", "739_workspace.pkl", "595_workspace.pkl", "317_workspace.pkl", "887_workspace.pkl", "490_workspace.pkl", "20_workspace.pkl", "429_workspace.pkl", "31_workspace.pkl", "686_workspace.pkl", "332_workspace.pkl", "260_workspace.pkl", "995_workspace.pkl", "350_workspace.pkl", "741_workspace.pkl", "329_workspace.pkl", "983_workspace.pkl", "458_workspace.pkl", "569_workspace.pkl", "8_workspace.pkl", "991_workspace.pkl", "754_workspace.pkl", "198_workspace.pkl", "442_workspace.pkl", "566_workspace.pkl", "897_workspace.pkl", "92_workspace.pkl", "629_workspace.pkl", "708_workspace.pkl", "478_workspace.pkl", "28_workspace.pkl", "387_workspace.pkl", "990_workspace.pkl", "288_workspace.pkl", "743_workspace.pkl", "842_workspace.pkl", "138_workspace.pkl", "588_workspace.pkl", "378_workspace.pkl", "543_workspace.pkl", "815_workspace.pkl", "759_workspace.pkl", "625_workspace.pkl", "286_workspace.pkl", "602_workspace.pkl", "589_workspace.pkl", "459_workspace.pkl", "817_workspace.pkl", "653_workspace.pkl", "509_workspace.pkl", "928_workspace.pkl", "243_workspace.pkl", "823_workspace.pkl", "441_workspace.pkl", "124_workspace.pkl", "266_workspace.pkl", "274_workspace.pkl", "302_workspace.pkl", "967_workspace.pkl", "5_workspace.pkl", "870_workspace.pkl", "474_workspace.pkl", "830_workspace.pkl", "397_workspace.pkl", "929_workspace.pkl", "635_workspace.pkl", "34_workspace.pkl", "290_workspace.pkl", "139_workspace.pkl", "505_workspace.pkl", "493_workspace.pkl", "171_workspace.pkl", "43_workspace.pkl", "921_workspace.pkl", "950_workspace.pkl", "781_workspace.pkl", "49_workspace.pkl", "340_workspace.pkl", "646_workspace.pkl", "562_workspace.pkl", "599_workspace.pkl", "866_workspace.pkl", "323_workspace.pkl", "857_workspace.pkl", "649_workspace.pkl", "173_workspace.pkl", "559_workspace.pkl", "664_workspace.pkl", "568_workspace.pkl", "745_workspace.pkl", "886_workspace.pkl", "615_workspace.pkl", "312_workspace.pkl", "23_workspace.pkl", "687_workspace.pkl", "163_workspace.pkl", "427_workspace.pkl", "517_workspace.pkl", "221_workspace.pkl", "419_workspace.pkl", "518_workspace.pkl", "851_workspace.pkl", "79_workspace.pkl", "186_workspace.pkl", "264_workspace.pkl", "577_workspace.pkl", "430_workspace.pkl", "924_workspace.pkl", "311_workspace.pkl", "888_workspace.pkl", "421_workspace.pkl", "93_workspace.pkl", "316_workspace.pkl", "144_workspace.pkl", "227_workspace.pkl", "920_workspace.pkl", "120_workspace.pkl", "488_workspace.pkl", "106_workspace.pkl", "177_workspace.pkl", "644_workspace.pkl", "179_workspace.pkl", "847_workspace.pkl", "446_workspace.pkl", "47_workspace.pkl", "992_workspace.pkl", "652_workspace.pkl", "172_workspace.pkl", "710_workspace.pkl", "131_workspace.pkl", "840_workspace.pkl", "147_workspace.pkl", "325_workspace.pkl", "808_workspace.pkl", "194_workspace.pkl", "696_workspace.pkl", "471_workspace.pkl", "814_workspace.pkl", "501_workspace.pkl", "884_workspace.pkl", "514_workspace.pkl", "584_workspace.pkl", "744_workspace.pkl", "916_workspace.pkl", "510_workspace.pkl", "604_workspace.pkl", "973_workspace.pkl", "466_workspace.pkl", "984_workspace.pkl", "275_workspace.pkl", "904_workspace.pkl", "205_workspace.pkl", "711_workspace.pkl", "732_workspace.pkl", "355_workspace.pkl", "587_workspace.pkl", "875_workspace.pkl", "17_workspace.pkl", "42_workspace.pkl", "451_workspace.pkl", "109_workspace.pkl", "127_workspace.pkl", "767_workspace.pkl", "77_workspace.pkl", "529_workspace.pkl", "39_workspace.pkl", "837_workspace.pkl", "742_workspace.pkl", "974_workspace.pkl", "859_workspace.pkl", "196_workspace.pkl", "388_workspace.pkl", "707_workspace.pkl", "268_workspace.pkl", "439_workspace.pkl", "896_workspace.pkl", "52_workspace.pkl", "654_workspace.pkl", "606_workspace.pkl", "89_workspace.pkl", "714_workspace.pkl", "4_workspace.pkl", "856_workspace.pkl", "500_workspace.pkl", "170_workspace.pkl", "891_workspace.pkl", "404_workspace.pkl", "94_workspace.pkl", "645_workspace.pkl", "149_workspace.pkl", "519_workspace.pkl", "24_workspace.pkl", "614_workspace.pkl", "674_workspace.pkl", "15_workspace.pkl", "576_workspace.pkl", "620_workspace.pkl", "414_workspace.pkl", "225_workspace.pkl", "62_workspace.pkl", "942_workspace.pkl", "386_workspace.pkl", "242_workspace.pkl", "878_workspace.pkl", "32_workspace.pkl", "306_workspace.pkl", "530_workspace.pkl", "117_workspace.pkl", "675_workspace.pkl", "582_workspace.pkl", "115_workspace.pkl", "579_workspace.pkl", "470_workspace.pkl", "701_workspace.pkl", "338_workspace.pkl", "877_workspace.pkl", "705_workspace.pkl", "190_workspace.pkl", "899_workspace.pkl", "69_workspace.pkl", "885_workspace.pkl", "175_workspace.pkl", "554_workspace.pkl", "300_workspace.pkl", "58_workspace.pkl", "230_workspace.pkl", "219_workspace.pkl", "824_workspace.pkl", "709_workspace.pkl", "224_workspace.pkl", "11_workspace.pkl", "426_workspace.pkl", "157_workspace.pkl", "628_workspace.pkl", "453_workspace.pkl", "431_workspace.pkl", "233_workspace.pkl", "913_workspace.pkl", "238_workspace.pkl", "852_workspace.pkl", "697_workspace.pkl", "263_workspace.pkl", "37_workspace.pkl", "622_workspace.pkl", "889_workspace.pkl", "735_workspace.pkl", "354_workspace.pkl", "56_workspace.pkl", "166_workspace.pkl", "676_workspace.pkl", "932_workspace.pkl", "462_workspace.pkl", "202_workspace.pkl", "968_workspace.pkl", "455_workspace.pkl", "892_workspace.pkl", "963_workspace.pkl", "125_workspace.pkl", "848_workspace.pkl", "113_workspace.pkl", "158_workspace.pkl", "259_workspace.pkl", "285_workspace.pkl", "618_workspace.pkl", "680_workspace.pkl", "650_workspace.pkl", "659_workspace.pkl", "560_workspace.pkl", "223_workspace.pkl", "351_workspace.pkl", "180_workspace.pkl", "322_workspace.pkl", "597_workspace.pkl", "25_workspace.pkl", "879_workspace.pkl", "962_workspace.pkl", "882_workspace.pkl", "642_workspace.pkl", "760_workspace.pkl", "151_workspace.pkl", "966_workspace.pkl", "103_workspace.pkl", "108_workspace.pkl", "27_workspace.pkl", "586_workspace.pkl", "525_workspace.pkl", "240_workspace.pkl", "802_workspace.pkl", "828_workspace.pkl", "790_workspace.pkl", "555_workspace.pkl", "871_workspace.pkl", "195_workspace.pkl", "795_workspace.pkl", "155_workspace.pkl", "961_workspace.pkl", "667_workspace.pkl", "515_workspace.pkl", "357_workspace.pkl", "310_workspace.pkl", "129_workspace.pkl", "444_workspace.pkl", "295_workspace.pkl", "492_workspace.pkl", "220_workspace.pkl", "418_workspace.pkl", "440_workspace.pkl", "142_workspace.pkl", "345_workspace.pkl", "346_workspace.pkl", "280_workspace.pkl", "211_workspace.pkl", "610_workspace.pkl", "428_workspace.pkl", "3_workspace.pkl", "261_workspace.pkl", "938_workspace.pkl", "364_workspace.pkl", "296_workspace.pkl", "672_workspace.pkl", "244_workspace.pkl", "747_workspace.pkl", "948_workspace.pkl", "556_workspace.pkl", "598_workspace.pkl", "145_workspace.pkl", "21_workspace.pkl", "271_workspace.pkl", "593_workspace.pkl", "59_workspace.pkl", "265_workspace.pkl", "880_workspace.pkl", "379_workspace.pkl", "655_workspace.pkl", "457_workspace.pkl", "895_workspace.pkl", "456_workspace.pkl", "959_workspace.pkl", "666_workspace.pkl", "975_workspace.pkl", "137_workspace.pkl", "507_workspace.pkl", "333_workspace.pkl", "71_workspace.pkl", "152_workspace.pkl", "370_workspace.pkl", "864_workspace.pkl", "452_workspace.pkl", "448_workspace.pkl", "98_workspace.pkl", "668_workspace.pkl", "215_workspace.pkl", "184_workspace.pkl", "792_workspace.pkl", "748_workspace.pkl", "822_workspace.pkl", "1_workspace.pkl", "41_workspace.pkl", "26_workspace.pkl", "390_workspace.pkl", "487_workspace.pkl", "972_workspace.pkl", "728_workspace.pkl", "362_workspace.pkl", "917_workspace.pkl", "935_workspace.pkl", "73_workspace.pkl", "401_workspace.pkl", "335_workspace.pkl", "234_workspace.pkl", "865_workspace.pkl", "336_workspace.pkl", "533_workspace.pkl", "811_workspace.pkl", "282_workspace.pkl", "420_workspace.pkl", "403_workspace.pkl", "183_workspace.pkl", "722_workspace.pkl", "516_workspace.pkl", "122_workspace.pkl", "182_workspace.pkl", "766_workspace.pkl", "954_workspace.pkl", "539_workspace.pkl", "762_workspace.pkl", "413_workspace.pkl", "563_workspace.pkl", "740_workspace.pkl", "819_workspace.pkl", "538_workspace.pkl", "188_workspace.pkl", "939_workspace.pkl", "540_workspace.pkl", "636_workspace.pkl", "82_workspace.pkl", "128_workspace.pkl", "294_workspace.pkl", "484_workspace.pkl", "716_workspace.pkl", "684_workspace.pkl", "578_workspace.pkl", "838_workspace.pkl", "648_workspace.pkl", "941_workspace.pkl", "247_workspace.pkl", "660_workspace.pkl", "826_workspace.pkl", "506_workspace.pkl", "980_workspace.pkl", "971_workspace.pkl", "656_workspace.pkl", "372_workspace.pkl", "498_workspace.pkl", "436_workspace.pkl", "148_workspace.pkl", "450_workspace.pkl", "634_workspace.pkl", "640_workspace.pkl", "153_workspace.pkl", "720_workspace.pkl", "611_workspace.pkl", "143_workspace.pkl", "432_workspace.pkl", "783_workspace.pkl", "590_workspace.pkl", "331_workspace.pkl", "699_workspace.pkl", "706_workspace.pkl", "773_workspace.pkl", "193_workspace.pkl", "801_workspace.pkl", "494_workspace.pkl", "861_workspace.pkl", "613_workspace.pkl", "845_workspace.pkl", "849_workspace.pkl", "949_workspace.pkl", "688_workspace.pkl", "970_workspace.pkl", "391_workspace.pkl", "289_workspace.pkl"]

# number_of_samples_per_workspace = 50
# samples_per_file = 10
# threads = 10
# results_dir = 'supervised_data_vision_temp_to_delete'
# scenario = 'vision_harder_small'

number_of_samples_per_workspace = 4000
samples_per_file = 1000
# threads = 100
threads = 10
results_dir = 'supervised_data_vision_harder'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

params_dir = os.path.abspath(os.path.expanduser('~/ModelBasedDDPG/scenario_params/{}/'.format(scenario)))
image_cache = ImageCache(params_dir)
collection_queries = []
workspace_ids = []
for cache_item in image_cache.items.values():
    if worksapces_to_query is None or cache_item.workspace_id in worksapces_to_query:
        collection_queries.extend(
            [(cache_item.workspace_id, cache_item.full_filename)] * number_of_samples_per_workspace)
        workspace_ids.append(cache_item.workspace_id)

data_collector = VisionRandomStepDataCollector(config, threads, query_parameters=collection_queries)
collected = 0

aa = datetime.datetime.now()

params_ids_to_tuples = {workspace_id: [] for workspace_id in workspace_ids}
params_ids_to_offset = {workspace_id: 0 for workspace_id in workspace_ids}
while collected < len(collection_queries):
    a = datetime.datetime.now()
    current_buffer = data_collector.generate_samples(samples_per_file)
    b = datetime.datetime.now()
    print 'data collection took: {}'.format(b - a)
    print_status_dist(current_buffer)

    for t in current_buffer:
        workspace_id = t[0]
        real_tuple = t[1:]
        params_ids_to_tuples[workspace_id].append(real_tuple)

    collected += len(current_buffer)

    for workspace_id in params_ids_to_tuples:
        workspace_buffer = params_ids_to_tuples[workspace_id]
        if len(workspace_buffer) >= samples_per_file:
            current_buffer = workspace_buffer[:samples_per_file]
            current_offset = params_ids_to_offset[workspace_id]
            dump_path = os.path.join(results_dir, '{}_{}.pkl'.format(workspace_id, current_offset))
            compressed_file = bz2.BZ2File(dump_path, 'w')
            pickle.dump(current_buffer, compressed_file)
            compressed_file.close()
            params_ids_to_offset[workspace_id] = current_offset + samples_per_file
            params_ids_to_tuples[workspace_id] = params_ids_to_tuples[workspace_id][samples_per_file:]
        assert len(params_ids_to_tuples[workspace_id]) < samples_per_file

bb = datetime.datetime.now()
print 'collection took: {}'.format(bb - aa)

for workspace_id in params_ids_to_tuples:
    assert len(params_ids_to_tuples[workspace_id]) == 0

data_collector.end()
