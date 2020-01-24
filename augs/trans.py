import torch

class Translator():
	def __init__(self):
		self.device = (torch.device('cuda:0' if torch.cuda.is_available() 
					   else 'cpu'))
		self.en2de = torch.hub.load(
			'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', 
			tokenizer='moses', bpe='fastbpe').to(self.device)
		self.de2en = torch.hub.load(
			'pytorch/fairseq', 'transformer.wmt19.de-en.single_model',
			tokenizer='moses', bpe='fastbpe').to(self.device)
		# self.en2ru = torch.hub.load(
		# 	'pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', 
		# 	tokenizer='moses', bpe='fastbpe').to(self.device)
		# self.ru2en = torch.hub.load(
		# 	'pytorch/fairseq', 'transformer.wmt19.ru-en.single_model',
		# 	tokenizer='moses', bpe='fastbpe').to(self.device)

	def aug(self, example, pivot):
		if pivot == 'de':
			return self.de2en.translate(
				self.en2de.translate(example, sampling=True, temperature=0.8),
				sampling=True, temperature=0.8)
		# elif pivot == 'ru':
		# 	return self.ru2en.translate(self.en2ru.translate(example))
		else:
			raise ValueError('Unrecognized pivot language.')