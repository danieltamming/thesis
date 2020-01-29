import os

from torch.utils.data import DataLoader

from dataset import BertDataset, RnnDataset
from managers.parent import DatasetManager
from utils.data import read_no_aug, read_trans_aug, partition_within_classes

class SubjDatasetManager(DatasetManager):
	def __init__(self, *args, **kwargs):
		super().__init__(get_subj, *args, **kwargs)


def get_subj(input_length, aug_mode):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	data_path = os.path.join(data_parent, 'subj')
	if aug_mode is None or aug_mode == 'synonym':
		file_path = os.path.join(data_path,'subj.txt')
		all_data = read_no_aug(file_path, input_length, True)
		# let 10% of data be the development set
		# dev_data, train_data = partition_within_classes(all_data, 0.1, False)
		# return {'dev': dev_data, 'train': train_data}
	elif aug_mode == 'trans':
		aug_file_path = os.path.join(data_path, 'trans_aug/subj.txt')
		all_data = read_trans_aug(aug_file_path)
	else:
		raise ValueError('Unrecognized augmentation.')
	dev_data, train_data = partition_within_classes(all_data, 0.1, False)
	return {'dev': dev_data, 'train': train_data}