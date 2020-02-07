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
	return int(line.split('small_label is ')[1].split(' ', 1)[0].rstrip(','))

def get_undersample(line):
	return line.split('undersample is ')[1].split(',', 1)[0] == 'True'

def get_small_prop(line):
	return float(line.split('small_prop is ')[1].split(',', 1)[0])

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

def plot_experiments():
	experiments = {}
	model = 'rnn'
	# model = 'bert'
	aug_mode = 'syn'
	# aug_mode = 'trans'
	data_name = 'sst'
	# data_name = 'subj'
	# filepath = 'logs/archived/bal_{}_{}_{}_pct.log'.format(model, aug_mode, data_name)
	filepath = 'logs/archived/bal_bert_trans_subj.log'
	with open(filepath) as f:
		line = f.readline()
		if 'RUN START' in line:
			line = f.readline()
		while line:
			tup = read_desc(line)
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
	averages = {tup: 100*mat.mean(0) for tup, mat in experiments.items()}
	for small_prop in sorted(list(set(key[3] for key in averages))):
		small_prop_averages = {key: avg for key, avg in averages.items()
							   if key[3] == small_prop}
		for small_label in sorted(list(set([key[2] for key in averages]))):
			# print(averages.keys())
			small_prop_label_averages = {key: avg for key, avg in small_prop_averages.items() 
										 if key[2] == small_label}
			oversample_avg = small_prop_label_averages[(-1, False, small_label, small_prop)]
			undersample_avg = small_prop_label_averages[(-1, True, small_label, small_prop)]
			del small_prop_label_averages[(-1, False, small_label, small_prop)]
			del small_prop_label_averages[(-1, True, small_label, small_prop)]
			for (geo, _, _, _), vec in sorted(small_prop_label_averages.items()):
				if geo not in [0.5, 0.6, 0.7] or small_prop == 1.0:
					continue
				# print('Geo: {}, label: {}, pct: {}'.format(geo, small_label, small_prop))
				plt.title('Rebalancing {} with {} after on {}% of label {}'
						  ' is kept.'.format(
						  		data_name, aug_mode, 
						  		100*small_prop, small_label))
				plt.ylabel('Validation Accuracy (%)')
				plt.xlabel('Training Epoch')
				# if data_name == 'sst':
				# 	plt.ylim((68, 84))
				# elif data_name == 'subj':
				# 	plt.ylim((80, 92))
				plt.plot(oversample_avg, label='oversampling', color='g', alpha=0.5)
				plt.plot(undersample_avg, label='undersampling', color='r', alpha=0.5)
				plt.plot(vec, label='geo {}'.format(geo), color='b', alpha=0.5)
				if model == 'rnn':
					plt.hlines(oversample_avg[25:].mean(), 0, 100, color='g')
					plt.hlines(undersample_avg[25:].mean(), 0, 100, color='r')
					plt.hlines(vec[25:].mean(), 0, 100, color='b')
				plt.legend()
				plt.show()

if __name__ == "__main__":
	plot_experiments()
	# with open('logs/main/seed_0_num_0.log') as f:
	# 	f.readline()
	# 	line = f.readline()
	# 	train_acc = []
	# 	val_acc = []
	# 	while line:
	# 		if is_training(line):
	# 			train_acc.append(get_acc(line))
	# 		elif is_validating(line):
	# 			val_acc.append(get_acc(line))
	# 		line = f.readline()
	# 	train_acc = 100*np.array(train_acc)
	# 	val_acc = 100*np.array(val_acc)
	# 	print(train_acc.max(), val_acc.max())
	# 	plt.plot(train_acc, label='training')
	# 	plt.plot(val_acc, label='validation')
	# 	plt.show()