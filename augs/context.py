import numpy as np
import random

def context_aug(seq, aug_dict, geo):
	num_to_replace = min(np.random.geometric(geo)-1, len(aug_dict))
	if num_to_replace == 0:
		return seq
	idxs = random.sample(aug_dict.keys(), num_to_replace)
	new_seq = []
	for i, eyedee in enumerate(seq):
		if i in idxs:
			syn_idx = min(np.random.geometric(geo), len(aug_dict[i])) - 1
			new_seq.append(aug_dict[i][syn_idx])
		else:
			new_seq.append(eyedee)
	assert len(seq) == len(new_seq)
	return new_seq