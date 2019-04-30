import numpy as np
import matplotlib.pyplot as plt

def single_loss_show(trace_len, loss_idx, name):
	p = plt.figure(figsize=(20,5))
	show_idx = [x for x in loss_idx if x in range(trace_len)]
	height = [1 for x in loss_idx if x in range(trace_len)]
	weight = [1 for x in loss_idx if x in range(trace_len)]
	plt.bar(show_idx, height, weight, color='black', edgecolor = 'none', label=name)

	# plt.bar(10000, 1, 0.1, color='black', label="packet loss")
	plt.legend(loc='upper right',fontsize=20)
	# plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	plt.axis([0, trace_len, 0, 1])
	plt.xticks(np.arange(0, trace_len + 1, trace_len/10))
	plt.tick_params(labelsize=20)
	# plt.yticks(np.arange(200, 1200+1, 200))
	plt.close()
	return p