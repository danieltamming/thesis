import os
import sys
import inspect
import random
import itertools
import string
from collections import Counter

from tqdm import tqdm
import torch

current_dir = os.path.dirname(
	os.path.abspath(inspect.getfile(inspect.currentframe()))
	)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from utils.parsing import get_device

def trans_aug(example, aug_counter, geo):
	# keep example with probability geo
	if random.random() < geo or not aug_counter:
		return example
	else:
		i = random.randrange(sum(aug_counter.values()))
		return next(itertools.islice(aug_counter.elements(), i, None))

# class Translator():
# 	def __init__(self):
# 		self.device = (torch.device(get_device() if torch.cuda.is_available() 
# 					   else 'cpu'))
# 		self.en2de = torch.hub.load(
# 			'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', 
# 			tokenizer='moses', bpe='fastbpe').to(self.device)
# 		self.de2en = torch.hub.load(
# 			'pytorch/fairseq', 'transformer.wmt19.de-en.single_model',
# 			tokenizer='moses', bpe='fastbpe').to(self.device)
# 		# self.en2ru = torch.hub.load(
# 		# 	'pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', 
# 		# 	tokenizer='moses', bpe='fastbpe').to(self.device)
# 		# self.ru2en = torch.hub.load(
# 		# 	'pytorch/fairseq', 'transformer.wmt19.ru-en.single_model',
# 		# 	tokenizer='moses', bpe='fastbpe').to(self.device)

# 	def aug(self, example, pivot):
# 		if pivot == 'de':
# 			return self.de2en.translate(
# 				self.en2de.translate(example, sampling=True, temperature=0.8),
# 				sampling=True, temperature=0.8)
# 		# elif pivot == 'ru':
# 		# 	return self.ru2en.translate(self.en2ru.translate(example))
# 		else:
# 			raise ValueError('Unrecognized pivot language.')

def is_english(s):
	try:
		s.encode(encoding='utf-8').decode('ascii')
	except UnicodeDecodeError:
		return False
	else:
		return True

def gen_trans_aug(example, en2de, de2en, beam, temperature, device):
	example_aug_list = []
	en_bin = en2de.encode(example).to(device)
	de_bin = en2de.generate(en_bin, beam=beam, sampling=True, 
							temperature=temperature)
	de_str_list = [en2de.decode(res['tokens']) for res in de_bin]
	for de_str in de_str_list:
		de_bin = de2en.encode(de_str)
		en_bin = de2en.generate(de_bin, beam=beam, sampling=True, 
								temperature=temperature)
		example_aug_list.extend([de2en.decode(res['tokens']) for res in en_bin])
	# return [s for s in example_aug_list if is_english(s)]
	return example_aug_list

def gen_save_trans(downloaded_dir, data_name, en2de, de2en, device):
	if data_name == 'sst':
		read_type = 'r'
	else:
		read_type = 'rb'
	data_dir = os.path.join(downloaded_dir, data_name)
	trans_aug_dir = os.path.join(data_dir, 'trans_aug')
	for filename in os.listdir(data_dir):
		no_aug_filepath = os.path.join(data_dir, filename)
		if not os.path.isfile(no_aug_filepath):
			continue
		trans_aug_filepath = os.path.join(trans_aug_dir, filename)
		with open(no_aug_filepath, read_type) as f, open(trans_aug_filepath, 'w') as g:
			for line in tqdm(f.read().splitlines()):
				if read_type == 'rb':
					line = line.decode('latin-1')
				label, example = line.split(maxsplit=1)
				g.write(label+'\n')
				g.write(example+'\n')
				example_aug_list = gen_trans_aug(example, en2de, de2en, 5, 0.8, device)
				safe = [s for s in example_aug_list if is_english(s)]
				unsafe = [s for s in example_aug_list if not is_english(s)]
				if len(safe) < 25:
					print(example)
					for s in unsafe:
						print(s)
				else:
					print('Success!')
				for example_aug, count in Counter(safe).most_common():
					g.write(str(count) + ' ' + example_aug + '\n')
				g.write('\n')

if __name__ == '__main__':
	device = torch.device(get_device() if torch.cuda.is_available() else 'cpu')
	en2de = torch.hub.load(
		'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', 
		tokenizer='moses', bpe='fastbpe').to(device)
	de2en = torch.hub.load(
		'pytorch/fairseq', 'transformer.wmt19.de-en.single_model',
		tokenizer='moses', bpe='fastbpe').to(device)

	downloaded_dir = '../DownloadedData/'
	# for data_name in ['sst', 'subj', 'trec']:
	for data_name in ['sst']:
		gen_save_trans(downloaded_dir, data_name, en2de, de2en, device)