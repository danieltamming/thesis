import os
import sys
import inspect

current_dir = os.path.dirname(
	os.path.abspath(inspect.getfile(inspect.currentframe()))
	)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (BertForMaskedLM, BertTokenizer, 
						  AdamW, get_linear_schedule_with_warmup)

from utils.data import get_sst, get_subj, get_trec
from utils.metrics import AverageMeter

class TextDataset(Dataset):
	def __init__(self, data_name, tokenizer, input_length):
		if data_name == 'sst':
			data = get_sst(None, None)['train']
		elif data_name == 'subj':
			data = get_subj(None, None)['train']
		elif data_name == 'trec':
			data = get_trec(None, None)['train']
		else:
			raise ValueError('Unrecognized data_name.')
		self.examples = []
		for (label, text, _) in data:
			tokenized_text = tokenizer.tokenize(text)
			ids_text = tokenizer.convert_tokens_to_ids(tokenized_text)
			input_ids = tokenizer.build_inputs_with_special_tokens(
				ids_text)
			input_ids = input_ids[:input_length]
			input_ids.extend((input_length - len(input_ids))
							 * [tokenizer.pad_token_id])
			self.examples.append((input_ids, label))
		# self.examples = make_length(self.examples, tokenizer, input_length)
	def __len__(self):
		return len(self.examples)

	def __getitem__(self, idx):
		'''
		May need to convert one or both elements of this tuple to torch tensor
		'''
		sequence, label = self.examples[idx]
		return torch.tensor(sequence), label

def mask_tokens(inputs, tokenizer):
	mlm_prob = 0.15
	targets = inputs.clone()
	prob_matrix = torch.full(inputs.shape, mlm_prob)
	special_tokens_mask = np.isin(inputs.numpy(), tokenizer.all_special_ids)
	prob_matrix.masked_fill_(torch.from_numpy(special_tokens_mask), value=0)
	masked_indices = torch.bernoulli(prob_matrix).bool()
	targets[~masked_indices] = -1
	# -------------------------------------------
	# new add for bug fix. check docs to see whether masked should be -1 or 0
	targets[~masked_indices] = 0
	# ----------------------------------------
	indices_replaced = (
		torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() 
		& masked_indices)
	inputs[indices_replaced] = tokenizer.mask_token_id
	indices_random = (
		torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() 
		& masked_indices 
		& ~indices_replaced)
	random_words = torch.randint(
		len(tokenizer), inputs.shape, dtype=torch.long)
	inputs[indices_random] = random_words[indices_random]
	return inputs, targets

def get_attention_mask(inputs):
	return (inputs > 0).float()

def cat_to_token_type(categories, seq_len):
	'''
	Returns batch_size x seq_len Tensor with values at [i,:] all
	equal to the category id. This provides the change from segment
	embeddings to category embeddings - thus chaning from contextual
	augmentation to conditional contextual augmentation
	'''
	return torch.ger(categories, torch.ones(seq_len, dtype=torch.long))

def mask_gen(seq, tokenizer):
	for i in range(seq.shape[0]):
		num = seq[i].item()
		if num in tokenizer.all_special_ids:
			continue
		seq[i] = tokenizer.mask_token_id
		yield i, seq
		seq[i] = num

class BertAgent:
	def __init__(self, data_name):
		self.data_name = data_name
		# CHANGE INPUT LENGTH. MAYBE 50?
		self.input_length = 25
		self.batch_size = 32
		self.num_train_epochs = 5
		self.checkpoint = 'bert-base-uncased'
		self.output_dir = 'ckpt'
		self.weight_decay = 0.0
		self.max_grad_norm = 1.0
		self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
		self.model = BertForMaskedLM.from_pretrained(self.checkpoint)
		self.dataset = TextDataset(self.data_name, self.tokenizer, 
								   self.input_length)
		self.loader = DataLoader(self.dataset, self.batch_size, 
								 pin_memory=True, shuffle=True)
		self.device = (torch.device('cuda:0' if torch.cuda.is_available() 
					   else 'cpu'))
		self.model = self.model.to(self.device)

	def save_checkpoint(self):
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)
		self.model.save_pretrained(self.output_dir)
		self.tokenizer.save_pretrained(self.output_dir)

	def train(self):
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in self.model.named_parameters() if
						not any(nd in n for nd in no_decay)], 
			 'weight_decay': self.weight_decay},
			{'params': [p for n, p in self.model.named_parameters() if 
						any(nd in n for nd in no_decay)], 
			 'weight_decay': 0.0}
		]
		# choosing AdamW defaults
		self.optimizer = AdamW(optimizer_grouped_parameters)
		total_steps = len(self.loader) * self.num_train_epochs
		self.scheduler = get_linear_schedule_with_warmup(
			self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
		self.model.train()
		self.model.zero_grad()
		for epoch in range(self.num_train_epochs):
			epoch_loss = self.train_one_epoch()
			epoch_loss = round(epoch_loss, 5)
			print('Epoch {} | loss = {}'.format(epoch, epoch_loss))

	def train_one_epoch(self):
		loss_meter = AverageMeter()
		for batch, categories in tqdm(self.loader):
			token_type_ids = cat_to_token_type(categories, batch.shape[1])
			attention_mask = get_attention_mask(batch)
			inputs, targets = mask_tokens(batch, self.tokenizer)
			inputs = inputs.to(self.device)
			attention_mask = attention_mask.to(self.device)
			token_type_ids = token_type_ids.to(self.device)
			targets = targets.to(self.device)

			# print(inputs)
			# # print(attention_mask.shape)
			# # print(token_type_ids.shape)
			# print(targets)
			# exit()

			# ----------------------------------------
			# check that targets and mask is created correctly
			# do research on MLM setup and training
			# ----------------------------------------

			loss, prediction_scores = self.model(
				inputs, attention_mask=attention_mask, 
				token_type_ids=token_type_ids, masked_lm_labels=targets)

			# --------------------------------------------------------
			_, pred = torch.max(prediction_scores, 2)
			pred = torch.mul(pred, attention_mask.long())
			# --------------------------------------------------------
			loss_meter.update(loss.item())
			loss.backward()
			nn.utils.clip_grad_norm_(self.model.parameters(), 
									 self.max_grad_norm)
			self.optimizer.step()
			self.scheduler.step()
			self.model.zero_grad()
		return loss_meter.val

	def develop(self):
		'''
		For testing and debugging the code
		'''
		# augment_loader = DataLoader(self.dataset, 1)
		# for seq, cat in tqdm(augment_loader):
		count = 0
		for seq, cat in self.dataset:
			count += 1
			print(cat)
			print(self.tokenizer.convert_ids_to_tokens(seq.tolist()))
			for i, masked_seq in mask_gen(seq, self.tokenizer):
				print(i)
				batch = torch.unsqueeze(masked_seq, 0)
				token_type_ids = cat*torch.ones(batch.shape[1], 
												dtype=torch.long)
				# token_type_ids = cat_to_token_type(cat, batch.shape[1])
				attention_mask = get_attention_mask(batch)
				batch = batch.to(self.device)
				attention_mask = attention_mask.to(self.device)
				token_type_ids = token_type_ids.to(self.device)
				prediction_scores = self.model(
					batch, attention_mask=attention_mask, 
					token_type_ids=token_type_ids)[0].data
				ith_preds = prediction_scores[0,i,:]
				# ith_pred = F.softmax(ith_pred, 0)
				# top_5_vals, top_5_idxs = torch.topk(ith_preds, 5)
				top_10_ids = torch.topk(ith_preds, 10)[1].tolist()
				print(self.tokenizer.convert_ids_to_tokens(top_10_ids))
				print()
				# plt.hist(ith_pred.numpy())
				# plt.show()
				# exit()
				# _, pred = torch.max(prediction_scores, 2)
				# pred = torch.mul(pred, attention_mask.long())
				# print(pred)
				# first_seq = pred[0,...].tolist()
				# print(self.tokenizer.convert_ids_to_tokens(first_seq)[i])
				# print()
			if count > 5:
				exit()

	def augment(self):
		data = []
		for seq, cat in tqdm(self.dataset):
			aug = {}
			for i, masked_seq in mask_gen(seq, self.tokenizer):
				batch = torch.unsqueeze(masked_seq, 0)
				token_type_ids = cat*torch.ones(batch.shape[1], 
												dtype=torch.long)
				# token_type_ids = cat_to_token_type(cat, batch.shape[1])
				attention_mask = get_attention_mask(batch)
				batch = batch.to(self.device)
				attention_mask = attention_mask.to(self.device)
				token_type_ids = token_type_ids.to(self.device)
				prediction_scores = self.model(
					batch, attention_mask=attention_mask, 
					token_type_ids=token_type_ids)[0].data
				ith_preds = prediction_scores[0,i,:]
				top_10_ids = torch.topk(ith_preds, 10)[1].tolist()
				# top_10_toks = self.tokenizer.convert_ids_to_tokens(top_10_ids)

				# changed to shift aug dict, since follwing line shifts seq by 1
				# aug[i] = top_10_ids
				aug[i-1] = top_10_ids
			# remove all padding and other special token ids
			clean_seq = [eyedee for eyedee in seq.tolist() if eyedee 
						 not in self.tokenizer.all_special_ids]
			assert clean_seq == seq.tolist()[1:1+len(clean_seq)]
			# -----------------------------------
			s = self.tokenizer.decode(clean_seq)
			flag = False
			for i in range(len(s)-3):
				if all(s[i] == c for c in s[i+1:i+3]):
					flag = True
					print(s)
			assert not flag
			# -----------------------------------

			data.append((clean_seq, cat, aug))

		# target_filename = 'data/aug_bert_ids.pickle'
		with open(self.aug_data_filename, 'wb') as f:
			pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
	data_name = 'sst'
	# data_name = 'subj'
	# data_name = 'trec'
	agent = BertAgent(data_name)
	agent.train()
	agent.save_checkpoint()
	# agent.develop()
	# agent.augment()


'''
READY TO RUN TRAIN THEN AUGMENT AGAIN
check for error with too many letters in a row


NEED TO ADD SCHEDULER IN ORDER TO GET TRAINING

'''