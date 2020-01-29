import os

from torch.utils.data import DataLoader

from dataset import BertDataset, RnnDataset
from managers.parent import DatasetManager
from utils.data import partition_within_classes, read_trans_aug

class TrecDatasetManager(DatasetManager):
	def __init__(self, *args, **kwargs):
		super().__init__(get_trec, *args, **kwargs)


def get_trec(input_length, aug_mode):
	if aug_mode is None or aug_mode == 'synonym':
		script_path = os.path.dirname(os.path.realpath(__file__))
		repo_path = os.path.join(script_path, os.pardir)
		data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
		target_parent = os.path.join(repo_path, 'data')
		data_path = os.path.join(data_parent,'trec')
		data_dict = {}
		for set_name in ['train', 'test']:
			set_path = os.path.join(data_path, set_name+'.txt')
			set_data = []
			with open(set_path, 'rb') as f:
				for line in f.read().splitlines():
					line = line.decode('latin-1')
					label, example = line.split(' ', 1)
					label = int(label)
					example = ' '.join(example.split()[:input_length])
					set_data.append((label, example, None))
			data_dict[set_name] = set_data
		# split given training split into training and dev set
		dev_data, train_data = partition_within_classes(data_dict['train'], 0.1, False)
		data_dict['dev'] = dev_data
		data_dict['train'] = train_data
		return data_dict
	# elif aug_mode == 'trans':
	# 	