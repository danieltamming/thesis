import os

from torch.utils.data import DataLoader

from dataset import BertDataset, RnnDataset
from managers.parent import DatasetManager
from utils.data import partition_within_classes

class SubjDatasetManager(DatasetManager):
	def __init__(self, *args, **kwargs):
		super().__init__(get_subj, *args, **kwargs)


def get_subj(input_length):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	target_parent = os.path.join(repo_path, 'data')
	file_path = os.path.join(data_parent,'subj/subj.txt')
	all_data = []
	with open(file_path, 'rb') as f:
		for line in f.read().splitlines():
			line = line.decode('latin-1')
			label, example = line.split(maxsplit=1)
			label = int(label)
			example = ' '.join(example.split()[:input_length])
			all_data.append((label, example))
	# let 10% of data be the development set
	dev_data, train_data = partition_within_classes(all_data, 0.1, False)
	return {'dev': dev_data, 'train': train_data}
