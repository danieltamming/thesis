from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def is_training(line):
	return line.split(' ', 1)[0].endswith('Training')

def is_validating(line):
	return line.split(' ', 1)[0].endswith('Validating')

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

def read_desc(line):
	aug_mode = get_aug_mode(line)
	small_label = get_small_label(line)
	small_prop = get_small_prop(line)
	if aug_mode == 'None':
		undersample = get_undersample(line)
		return (-1, undersample, small_label, small_prop)
	else:
		geo = float(line.split('geo is ')[1].split(maxsplit=1)[0].rstrip(','))
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
			tup = read_desc(line)
			if avg_across_labels:
				tup = (tup[0], tup[1], 'both', tup[3]) # ignore small_label
			accs = []
			line = f.readline()
			while is_training(line) or is_validating(line):
				if is_validating(line):
					accs.append(get_acc(line))
				line = f.readline()
			if tup not in experiments:
				experiments[tup] = np.array(accs)
			else:
				experiments[tup] = np.vstack([experiments[tup], np.array(accs)])
	return experiments

def plot_experiments():
	avg_across_labels = True
	model = 'rnn'
	# model = 'bert'
	# aug_mode = 'syn'
	aug_mode = 'trans'
	data_name = 'sst'
	# data_name = 'subj'
	filepath = 'logs/archived/bal_{}_{}_{}_pct.log'.format(model, aug_mode, data_name)
	# filepath = 'logs/archived/bal_bert_trans_sst.log'
	# filepath = 'logs/archived/bal_bert_trans_subj.log'
	experiments = read_experiments(filepath, avg_across_labels)
	
	averages = experiments

	for small_prop in sorted(list(set(key[3] for key in averages))):
		small_prop_averages = {key: avg for key, avg in averages.items()
							   if key[3] == small_prop}
		for small_label in sorted(list(set([key[2] for key in averages]))):
			# if small_label == 1:
			# 	continue
			# print(averages.keys())
			small_prop_label_averages = {key: avg for key, avg in small_prop_averages.items() 
										 if key[2] == small_label}
			oversample_avg = small_prop_label_averages[(-1, False, small_label, small_prop)]
			undersample_avg = small_prop_label_averages[(-1, True, small_label, small_prop)]
			del small_prop_label_averages[(-1, False, small_label, small_prop)]
			del small_prop_label_averages[(-1, True, small_label, small_prop)]

			colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
					  'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 
					  'tab:olive', 'tab:cyan']
			for ((geo, _, _, _), vec), color in zip(sorted(small_prop_label_averages.items()), colors):
			# for (geo, _, _, _), vec in sorted(small_prop_label_averages.items()):
				if small_prop == 1.0:
					continue
				if geo not in [0.5, 0.6, 0.7, 0.8, 0.9]:
					continue
				plt.title('Rebalancing {} with {} after on {}% of label {}'
						  ' is kept.'.format(
						  		data_name, aug_mode, 
						  		100*small_prop, small_label))
				plt.ylabel('Validation Accuracy (%)')
				plt.xlabel('Training Epoch')
				# plot_mat(100*oversample_avg, True, label='oversampling', color='g', alpha=0.5)
				# plot_mat(100*undersample_avg, True, label='undersampling', color='r', alpha=0.5)
				# plot_mat(100*vec, True, label='geo {}'.format(geo), color='b', alpha=0.5)
				# plt.legend()
				# plt.show()

				plot_mat(100*vec, False, label='geo {}'.format(geo), color=color, alpha=0.5)
			if small_prop != 1.0:
				plot_mat(100*oversample_avg, False, label='oversampling', color='g', alpha=0.5)
				plot_mat(100*undersample_avg, False, label='undersampling', color='r', alpha=0.5)
				plt.legend()
				plt.show()

if __name__ == "__main__":
	plot_experiments()
	exit()

	avgs = []
	for name in ['first.log', 'second.log']:
	# for name in ['third.log']:
		with open('logs/main/' + name) as f:
			f.readline()
			line = f.readline()
			accs = []
			all_accs = []
			while line:
				if is_training(line):
					pass
				elif is_validating(line):
					accs.append(get_acc(line))
				else:
					# f.readline()
					all_accs.append(np.array(accs))
					accs = []
				line = f.readline()
		all_accs = np.vstack(all_accs)
		avgs.append(all_accs.mean(0))
	plt.plot(avgs[0], label='first', color='b')
	plt.plot(avgs[1], label='second', color='g')
	print(avgs[0].max())
	# print(avgs[1].max())
	plt.legend()
	plt.show()