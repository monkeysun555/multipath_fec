import numpy as np
import scipy.sparse.linalg as sla
import show
import os

RANDOM_SEED = 20
RAND_RANGE = 1000
IS_BURSTY = 1
IS_MULTIPATH = 0
IS_PLOT = 0
# Assuming do FEC before multipath
NON_BURSTY_MAT = [[0.906, 0.094, 0, 0, 0],
				[0.97, 0, 0.03, 0, 0],
				[1, 0, 0, 0, 0],
				[1, 0, 0, 0, 0],
				[1, 0, 0, 0, 0]]

BURSTY_MAT = [[0.92, 0.08, 0, 0, 0],
			[0.89, 0, 0.11, 0, 0],
			[0.95, 0, 0, 0.05, 0],
			[0.97, 0, 0, 0, 0.03],
			[1.0, 0, 0, 0, 0]]
			
SIM_LEN = 20000
if not IS_PLOT:
	TESTING_LEN = 200000
else:
	TESTING_LEN = 500
ORI_SIZE = 4
FEC_SIZE = 2
MODE = 0	# 0 for single path, 1 for multipath
MULTI_SHOW_LEN = 200
FIGS_FILE = './figs/'

def fec_encoding():
	p_id = 0
	pkts_trace = []
	total_n_fecs = 0
	while p_id < SIM_LEN:
		n_ori_pkts = np.minimum(ORI_SIZE, SIM_LEN - p_id)
		n_fec = FEC_SIZE
		for _ in range(n_ori_pkts):
			pkts_trace.append(p_id)
			p_id += 1
		fec_title = str(pkts_trace[-1])
		for i in range(n_fec):
			pkts_trace.append(fec_title + '_fec_' + str(i))
			total_n_fecs += 1
	return pkts_trace, total_n_fecs

def fec_decoding(trace):
	decoded_trace = []
	n_unrecoved = 0
	n_groups = int(SIM_LEN/ORI_SIZE)
	n_left_pkts = SIM_LEN - n_groups*ORI_SIZE
	recv_idx = 0
	# print trace
	for group_idx in range(n_groups):
		s_id = group_idx * ORI_SIZE
		e_id = s_id + ORI_SIZE - 1
		pre_idx = recv_idx
		while recv_idx < len(trace):
			assert trace[recv_idx] >= s_id
			if not isinstance(trace[recv_idx], int):
				curr_p_id = int(trace[recv_idx][0])
			else:
				curr_p_id = trace[recv_idx]
			if curr_p_id > e_id:
				break
			recv_idx += 1
		curr_group = trace[pre_idx:recv_idx]
		# print curr_group
		group_idx += 1
		# Decoding
		if len(curr_group) >= ORI_SIZE:
			decoded_trace.extend(range(s_id,e_id+1))
		else:
			# decoded_trace.extend([x for x in curr_group if isinstance(x, int)])
			n_unrecoved += len([x for x in curr_group if isinstance(x, int)])
			print "Failed group:", range(s_id, e_id+1), ",", [x for x in curr_group if isinstance(x, int)], "is received"
	# Process the remainning packets
	recv_packets_last = trace[recv_idx:]
	if len(recv_packets_last) >= n_left_pkts:
		decoded_trace.extend(range(n_groups*ORI_SIZE, SIM_LEN))
	else:
		n_unrecoved += len([x for x in recv_packets_last if isinstance(x, int)])
		# decoded_trace.extend([x for x in recv_packets_last if isinstance(x, int)])
	# print decoded_trace
	return decoded_trace, n_unrecoved

def split_trace(trace):
	n_paths = 2
	s_traces = [[] for _ in range(n_paths)]
	i = 0
	while i < len(trace):
		s_traces[i%n_paths].append(trace[i])
		i += 1
	return s_traces

def single_path_deliver(trace, cum_e_vec, cum_mat):
	rec_trace = []
	curr_state = (cum_e_vec > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
	# Assuming start from state 0
	# curr_state = 0
	p_id = 0
	while p_id < len(trace):
		if curr_state == 0:
			rec_trace.append(trace[p_id])
			p_id += 1
		else:
			p_id += curr_state	# state represent num of lost pakcets
		curr_cum_prob = cum_mat[curr_state]
		curr_state = (curr_cum_prob > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
	return rec_trace

def ratio_testing(cum_e_vec, cum_mat):
	trace = range(TESTING_LEN)
	rec_trace = []
	curr_state = (cum_e_vec > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
	# Assuming start from state 0
	# curr_state = 0
	p_id = 0
	while p_id < len(trace):
		if curr_state == 0:
			rec_trace.append(trace[p_id])
			p_id += 1
		else:
			p_id += curr_state	# state represent num of lost pakcets
		curr_cum_prob = cum_mat[curr_state]
		curr_state = (curr_cum_prob > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
	loss_show = list(set(trace) - set(rec_trace))
	return (1.0 - float(len(rec_trace))/ TESTING_LEN)*100, loss_show

def combine_traces(traces):
	combined_trace = []
	# Pre process, get rid of 0 length recv trace:
	for i in range(len(traces)):
		if not len(traces[i]):
			del traces[i]

	while len(traces) > 0:
		next_group = []
		for i in range(len(traces)):
			# print traces[i]
			next_group.append(traces[i][0])

		for i in range(len(next_group)):
			if not isinstance(next_group[i], int):
				int_part = int(next_group[i].split('_')[0])
				float_part = float(next_group[i].split('_')[-1]) + 1
				# print int_part
				# print float_part
				next_group[i] = int_part + float_part/100
		next_pkt_idx = next_group.index(min(next_group))
		combined_trace.append(traces[next_pkt_idx].pop(0))
		if len(traces[next_pkt_idx]) == 0:
			del traces[next_pkt_idx]
		# print traces
	return combined_trace

def calculate_loss_of_path(send_t, recv_t):
	assert len(send_t) == len(recv_t)
	loss_rec = []
	loss_idx_plot = []
	ratios = []
	for i in range(len(send_t)):
		# print send_t[i]
		temp_loss_idx = []
		temp_loss = list(set(send_t[i]) - set(recv_t[i]))
		loss_rec.append(temp_loss)
		for loss in temp_loss:
			temp_loss_idx.append(send_t[i].index(loss))
		loss_idx_plot.append(temp_loss_idx)
		ratios.append(float(len(temp_loss))/len(send_t[i])*100)
	return loss_rec, loss_idx_plot, ratios

def EGM():
	if not os.path.isdir(FIGS_FILE):
		os.makedirs(FIGS_FILE)
	# Show Conf
	print "<===============>"
	if IS_MULTIPATH:
		print "Multipath Mode"
	else:
		print "Single Path Mode"
	if IS_BURSTY:
		print "In Bursty Loss Environment"
	else:
		print "In Non-bursty Loss Environment"
	np.random.seed(RANDOM_SEED)
	pkts_trace, total_n_fecs = fec_encoding()
	# print pkts_trace
	if IS_BURSTY:
		prob_mat = BURSTY_MAT
	else:
		prob_mat = NON_BURSTY_MAT
	e_val, e_vec = sla.eigs(np.array(prob_mat).T, k=1, which='LM')
	cum_e_vec = np.cumsum(e_vec)
	cum_mat = [np.cumsum(prob) for prob in prob_mat]
	exp_ratio, loss_show = ratio_testing(cum_e_vec, cum_mat)

	print "Steady Matrix:", e_vec
	print "Expected loss ratio (single path) is", exp_ratio, "%"
	print "<===============>"

	if IS_MULTIPATH:
		splited_traces = split_trace(pkts_trace)
		recv_traces = []
		# print splited_traces
		for trace in splited_traces[:]:
			recv_trace = single_path_deliver(trace, cum_e_vec, cum_mat)
			# if len(recv_trace) > 0:
			recv_traces.append(recv_trace)
		# Before combination, check the loss on each path
		loss_rec, loss_plot_real, ratio_rec = calculate_loss_of_path(splited_traces, recv_traces)
		plot_traces_len = [len(trace) for trace in recv_traces]
		rec_trace = combine_traces(recv_traces)

	else:
		rec_trace = single_path_deliver(pkts_trace, cum_e_vec, cum_mat)
	decoded_trace, n_wasted_pkts = fec_decoding(rec_trace)

	print SIM_LEN, "source packets and", total_n_fecs, "FEC packets,", SIM_LEN + total_n_fecs, "packets in total" 
	print "Overhead ratio is: ", float(total_n_fecs)/SIM_LEN
	print len(rec_trace), "packets are received."
	# print len(decoded_trace), "source packets are recoved"
	print len(decoded_trace) + n_wasted_pkts, "source packets are received"
	if IS_MULTIPATH:
		for i in range(len(loss_rec)):
			print "Packets lost on path", i, "is", len(loss_rec[i]), ", loss ratio is", ratio_rec[i]
	print "Total loss ratio is:", (1.0 - (float(len(decoded_trace)) + n_wasted_pkts)/SIM_LEN) * 100,"%"

	if IS_PLOT:
		figs = []
		names = []
		# Matrix evaluation
		name = "Matrix Loss_" + str(IS_BURSTY)
		mat_eva = show.single_loss_show(TESTING_LEN, loss_show, name)
		names.append(name)
		figs.append(mat_eva)

		# For multipath:
		if IS_MULTIPATH:
			for i in range(len(plot_traces_len)):
				name = "Multipath Loss Path" + str(i)
				p = show.single_loss_show(MULTI_SHOW_LEN, loss_plot_real[i], name)
				names.append(name)
				figs.append(p)

		for i in range(len(figs)):
			figs[i].savefig(FIGS_FILE + names[i] + '.eps', format='eps', dpi=1000, figsize=(30, 10))

	return 

def main():
	EGM()
	return 

if __name__ == '__main__':
	main()