import os

from torch.utils.data import DataLoader

from dataset import BertDataset, RnnDataset
from managers.parent import DatasetManager
from utils.data import partition_within_classes

class TrecDatasetManager(DatasetManager):
	def __init__(self, *args, **kwargs):
		super().__init__(get_trec, *args, **kwargs)


def get_trec(input_length):
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
				set_data.append((label, example))
		data_dict[set_name] = set_data
	# split given training split into training and dev set
	dev_data, train_data = partition_within_classes(data_dict['train'], 0.1)
	data_dict['dev'] = dev_data
	data_dict['train'] = train_data
	return data_dict


# class TrecDatasetManager:
# 	def __init__(self, config, model_type, input_length, 
# 				 aug_mode, pct_usage, geo, batch_size, nlp=None,
# 				 small_label=None, small_prop=None, balance_seed=None):
# 		self.config = config
# 		self.model_type = model_type
# 		self.input_length = input_length
# 		self.aug_mode = aug_mode
# 		self.pct_usage = pct_usage
# 		self.geo = geo
# 		self.batch_size = batch_size
# 		self.small_label = small_label
# 		self.small_prop = small_prop
# 		self.balance_seed = balance_seed
# 		self.data_dict = get_trec(self.input_length)

# 		if model_type == 'rnn':
# 			assert nlp is not None
# 			self.nlp = nlp

# 	def get_dev_ldrs(self):
# 		train_dataset = self.get_dataset('train')
# 		train_loader = DataLoader(
# 			train_dataset, self.batch_size, pin_memory=True, shuffle=True)
# 		val_dataset = self.get_dataset('dev')
# 		val_loader = DataLoader(
# 			val_dataset, self.batch_size, pin_memory=True)
# 		return train_loader, val_loader
	
# 	def get_dataset(self, split_key):
# 		if split_key == 'train':
# 			aug_mode = self.aug_mode
# 			geo = self.geo
# 			small_label = self.small_label
# 			small_prop = self.small_prop
# 		else:
# 			# val and test splits have no augmentation
# 			aug_mode = None
# 			geo = None
# 			small_label = None
# 			small_prop = None
# 		if self.model_type == 'bert':
# 			return BertDataset(self.data_dict[split_key], self.input_length, 
# 							  aug_mode, geo=geo, small_label=small_label, 
# 							  small_prop=small_prop,
# 							  balance_seed=self.balance_seed)
# 		elif self.model_type == 'rnn':
# 			return RnnDataset(self.data_dict[split_key], self.input_length, 
# 							  self.nlp, aug_mode, geo=geo, 
# 							  small_label=small_label, small_prop=small_prop,
# 							  balance_seed=self.balance_seed)
# 		else:
# 			raise ValueError('Unrecognized model type.')