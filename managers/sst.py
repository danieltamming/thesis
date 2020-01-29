import os
from collections import Counter

from torch.utils.data import DataLoader

from dataset import BertDataset, RnnDataset
from managers.parent import DatasetManager
from utils.data import read_trans_aug

class SSTDatasetManager(DatasetManager):
	def __init__(self, *args, **kwargs):
		super().__init__(get_sst, *args, **kwargs)


def get_sst(input_length, aug_mode):
	if aug_mode is None or aug_mode == 'synonym':
		script_path = os.path.dirname(os.path.realpath(__file__))
		repo_path = os.path.join(script_path, os.pardir)
		data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
		target_parent = os.path.join(repo_path, 'data')
		data_path = os.path.join(data_parent,'sst')
		data_dict = {}
		for set_name in ['train', 'dev', 'test']:
			set_path = os.path.join(data_path, set_name+'.txt')
			set_data = []
			with open(set_path) as f:
				for line in f.read().splitlines():
					label, example = line.split(maxsplit=1)
					label = int(label)
					example = ' '.join(example.split()[:input_length])
					set_data.append((label, example, None))
			data_dict[set_name] = set_data
		return data_dict
	elif aug_mode == 'trans':
		# not restricting to input length here but it isn't that necessary anyways
		# that offers more time savings in synonym as it restricts the search
		script_path = os.path.dirname(os.path.realpath(__file__))
		repo_path = os.path.join(script_path, os.pardir)
		data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
		target_parent = os.path.join(repo_path, 'data')
		data_path = os.path.join(data_parent,'sst/trans_aug')
		data_dict = {}
		for set_name in ['train', 'dev', 'test']:
			set_path = os.path.join(data_path, set_name+'.txt')
			data_dict[set_name] = read_trans_aug(set_path)
		return data_dict

# import random
# import itertools
# from collections import Counter
# from tqdm import tqdm
# def trans_aug(example, aug_counter, geo):
# 	# keep example with probability geo
# 	if random.random() < geo or not aug_counter:
# 		return example
# 	else:
# 		i = random.randrange(sum(aug_counter.values()))
# 		return next(itertools.islice(aug_counter.elements(), i, None))

# # train_data = get_sst(25, 'trans')['train']
# # labels = set([label for label, _, _ in train_data])
# # print(labels)

# exit()

# label, example, aug_counter = train_data[9]

# results = Counter()
# for _ in tqdm(range(10000000)):
# 	results[trans_aug(example, aug_counter, 0)] += 1

# for key in sorted(aug_counter.keys()):
# 	print(aug_counter[key]/sum(aug_counter.values()), results[key]/sum(results.values()))