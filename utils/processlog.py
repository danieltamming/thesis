from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

def is_training(line):
	return line.split(' ', 1)[0].endswith('Training')

def is_validating(line):
	return line.split(' ', 1)[0].endswith('Validating')

def get_loss(line):
	return float(line.split('loss: ')[-1].split(' ', 1)[0])

def get_acc(line):
	return float(line.split()[-1])

def get_aug_mode(line):
	return line.split('aug mode is ')[1].split(',', 1)[0]

def get_small_label(line):
	small_label = line.split('small_label is ')[1].split(' ', 1)[0].rstrip(',')
	if small_label == 'None':
		return None
	else:
		return int(small_label)

def get_undersample(line):
	return line.split('undersample is ')[1].split(',', 1)[0] == 'True'

def get_small_prop(line):
	small_prop = line.split('small_prop is ')[1].split(',', 1)[0]
	if small_prop == 'None':
		return None
	else:
		return float(small_prop)

def get_lr(line):
	lr = line.split('lr is ')[1].split(',', 1)[0]
	return float(lr)

def get_pct_usage(line):
	pct_usage = line.split('pct_usage is ')[1].split(',', 1)[0]
	return float(pct_usage)

def get_geo(line):
	geo = line.split('geo is ')[1].split(',', 1)[0]
	return float(geo)

def read_pct_desc(line):
	aug_mode = get_aug_mode(line)
	pct_usage = get_pct_usage(line)
	if aug_mode == 'None':
		return (-1, pct_usage, None)
	else:
		geo = get_geo(line)
		return (geo, pct_usage, aug_mode)

def read_imbalance_desc(line):
	aug_mode = get_aug_mode(line)
	small_label = get_small_label(line)
	small_prop = get_small_prop(line)
	if aug_mode == 'None':
		undersample = get_undersample(line)
		return (-1, undersample, small_label, small_prop)
	else:
		geo = get_geo(line)
		return (geo, None, small_label, small_prop)

def plot_mat(mat, err_bars, *args, **kwargs):
	avg = mat.mean(0)
	dev = mat.std(0)
	plt.plot(avg, *args, **kwargs)
	kwargs['alpha'] = kwargs['alpha']/5
	del kwargs['label']
	if err_bars:
		plt.fill_between(list(range(mat.shape[1])), 
						 avg-dev, avg+dev, *args, **kwargs)
	plt.plot(np.argmax(avg), avg.max(), kwargs['color'], marker='o')
	# plt.hlines(np.max(avg), 0, 100, color=kwargs['color'])

def read_experiments(filepath, avg_across_labels):
	experiments = {}
	with open(filepath) as f:
		line = f.readline()
		if 'RUN START' in line:
			line = f.readline()
		while line:
			tup = read_imbalance_desc(line)
			if avg_across_labels:
				tup = (tup[0], tup[1], 'both', tup[3]) # ignore small_label
			accs = []
			line = f.readline()
			while is_training(line) or is_validating(line):
				if is_validating(line):
					accs.append(get_acc(line))
				line = f.readline()
			# if len(accs) != 100:
			# 	continue
			if tup not in experiments:
				experiments[tup] = np.array(accs)
			else:
				experiments[tup] = np.vstack([experiments[tup], np.array(accs)])
	return experiments

def plot_imbalance_experiments():
	avg_across_labels = True
	model = 'rnn'
	# model = 'bert'
	# aug_mode = 'syn'
	# aug_mode = 'trans'
	aug_mode = 'context'
	data_name = 'sst'
	# data_name = 'subj'
	filepath = 'logs/archived/valids/bal_{}_{}_{}.log'.format(model, aug_mode, data_name)
	# filepath = 'logs/archived/bal_rnn_context_odds_10seeds.log'
	# filepath = 'logs/archived/older/bal_bert_trans_subj_pct.log'
	# filepath = 'logs/archived/older/bal_rnn_trans_subj_fine.log'
	err_bars = False
	experiments = read_experiments(filepath, avg_across_labels)
	
	averages = experiments

	for small_prop in sorted(list(set(key[3] for key in averages))):
		small_prop_averages = {key: avg for key, avg in averages.items()
							   if key[3] == small_prop}
		for small_label in sorted(list(set([key[2] for key in averages]))):
			small_prop_label_averages = {key: avg for key, avg in small_prop_averages.items() 
										 if key[2] == small_label}
			oversample_avg = small_prop_label_averages[(-1, False, small_label, small_prop)]
			undersample_avg = small_prop_label_averages[(-1, True, small_label, small_prop)]
			del small_prop_label_averages[(-1, False, small_label, small_prop)]
			del small_prop_label_averages[(-1, True, small_label, small_prop)]

			colors = ['tab:blue', 'tab:orange', 'tab:cyan', 'tab:red', 
					  'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 
					  'tab:olive', 'tab:green']
			for ((geo, _, _, _), vec), color in zip(sorted(small_prop_label_averages.items()), colors):
				if small_prop == 1.0:
					continue
				plt.title('Rebalancing {} with {} after on {}% of label {}'
						  ' is kept.'.format(
						  		data_name, aug_mode, 
						  		100*small_prop, small_label))
				plt.ylabel('Validation Accuracy (%)')
				plt.xlabel('Training Epoch')

				plot_mat(100*vec, err_bars, label='geo {}'.format(geo), color=color, alpha=0.5)
			if small_prop != 1.0:
				plot_mat(100*oversample_avg, err_bars, label='oversampling', color='g', alpha=0.5)
				plot_mat(100*undersample_avg, err_bars, label='undersampling', color='r', alpha=0.5)
				plt.legend()
				plt.show()

def get_num_epochs(line):
	num_epochs = line.split('max_epochs is ')[1].split(',', 1)[0]
	return int(num_epochs)

def plot_imbalance_tests():
	filepath = 'logs/archived/tests/test_bal_rnn_trans_subj.log'
	methods = ['Augmentation', 'Oversample', 'Undersample']
	df = pd.DataFrame(index=range(10, 100, 10))
	for col_name in methods:
		df[col_name] = [[] for _ in range(len(df))]
	with open(filepath) as f:
		line = f.readline()
		if 'RUN START' in line:
			line = f.readline()
		while line:
			small_prop = get_small_prop(line)
			undersample = get_undersample(line)
			aug_mode = get_aug_mode(line)
			geo = get_geo(line)
			if aug_mode == 'None':
				if undersample:
					mode = 'Undersample'
				else:
					mode = 'Oversample'
			else:
				mode = 'Augmentation'
			num_epochs = get_num_epochs(line)
			for _ in range(num_epochs):
				line = f.readline()
				if is_validating(line):
					line = f.readline()
			line = f.readline()
			# print(line)
			assert is_validating(line)
			df.loc[100*small_prop, mode].append(get_acc(line))
			line = f.readline()
	for col_name in methods:
		df[col_name+'_mean'] = df[col_name].apply(np.mean)
		df[col_name+'_std'] = df[col_name].apply(np.std)
	df = df.drop(columns=methods)
	for m in methods:
		sns.lineplot(x=df.index, y=m+'_mean', data=df, label=m)
		plt.fill_between(
			df.index, 
			df[m+'_mean'] - df[m+'_std'], 
			df[m+'_mean'] + df[m+'_std'],
			alpha=0.1
		)
		# plt.errorbar(x=df.index, y=df[m+'_mean'], yerr=df[m+'_std'])
	# sns.lineplot(
	# 	x=df.index, y='Augmentation_mean', data=df, label='Augmentation')
	# plt.fill_between(
	# 	df.index, 
	# 	df.Augmentation_mean - df.Augmentation_std, 
	# 	df.Augmentation_mean + df.Augmentation_std,
	# 	alpha=0.25
	# )
	# sns.lineplot(x=df.index, y='Undersample_mean', data=df, label='Undersample')
	# sns.lineplot(x=df.index, y='Oversample_mean', data=df, label='Oversample')
	plt.xlabel('Percentage of Minority Label Examples Left In Training Set')
	plt.ylabel('Accuracy (%)')
	plt.legend()
	plt.show()
	return df


if __name__ == "__main__":
	plot_imbalance_experiments()
	# plot_imbalance_tests()
	exit()
	# filepath = 'logs/main/seed_0_other_0.5_num_3.log'
	# experiments = {}
	# with open(filepath) as f:
	# 	line = f.readline()
	# 	if 'RUN START' in line:
	# 		line = f.readline()
	# 	while line:
	# 		tup = read_pct_desc(line)
	# 		accs = []
	# 		line = f.readline()
	# 		while is_training(line) or is_validating(line):
	# 			if is_validating(line):
	# 				accs.append(get_acc(line))
	# 			line = f.readline()
	# 		experiments[tup] = np.array(accs)
	# for tup, vec in experiments.items():
	# 	if tup[0] not in [0.3, 0.7]:
	# 		sns.lineplot(x=list(range(len(vec))), y=vec, label=tup[0])
	# plt.show()

# 0.4, 78
# 0.2, 79
# 0.3, 80
# 0.5, 80.5
# 0.3, 81
# 0.4, 81
# 0.7, 81.5
# 0.8, 81.5
# 0.9, 81.5