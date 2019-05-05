import numpy as np
import os
import scipy.sparse.linalg as sla
import show

SIM_LEN = 20000
IS_BURSTY = 1
IS_MULTIPATH = 0
if IS_MULTIPATH == 1:
	NUM_PATH = 2
IS_PLOT = 0

THROUGHPUT = 2000.0 	# Kbits per second
OL_PACKET_LEN = 1100		# Bytes
LINK_CAP = 100000.0		# Kbits per second
PROP_DELAY = 20.0 		# MS
QUEUING_DELAY = 3.0		# MS
AVE_PKT_LEN = 1000.0

N_PKTS_PER_S = THROUGHPUT * 1000 / 8 / OL_PACKET_LEN			
PKT_GAP_IN_MS = 1000.0 / N_PKTS_PER_S
LINK_PKTS_PER_S = LINK_CAP * 1000 / 8 / AVE_PKT_LEN
LINK_PKT_GAP_IN_MS = 1000.0 / LINK_PKTS_PER_S

RANDOM_SEED = 20
RAND_RANGE = 1000

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

if not IS_PLOT:
	TESTING_LEN = 200000
else:
	TESTING_LEN = 500
ORI_SIZE = 4
FEC_SIZE = 2
FEC_TIME = 1.0
FEC_DE_TIME = 1.0
MODE = 0	# 0 for single path, 1 for multipath
MULTI_SHOW_LEN = 200
FIGS_FILE = './figs/'

def split_trace(trace):
	n_paths = NUM_PATH
	s_traces = [[] for _ in range(n_paths)]
	i = 0
	while i < len(trace):
		s_traces[i%n_paths].append(trace[i])
		i += 1
	return s_traces

def deliver_pkts(trace):
	return np.random.exponential(LINK_PKT_GAP_IN_MS, len(trace))

def single_path_deliver(trace, cum_e_vec, cum_mat):
	rec_trace = []
	# deliver_time = deliver_pkts(trace)	

	# Init the first pkts
	trace[0].append(trace[0][2] + np.random.uniform(low=PROP_DELAY, high=PROP_DELAY+QUEUING_DELAY))
	for i in xrange(1, len(trace)):
		trace[i].append(np.maximum(trace[i-1][-1], trace[i][-1] + np.random.uniform(low=PROP_DELAY, high=PROP_DELAY+QUEUING_DELAY)))				# Insert the deliver time/received time to end,

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

def calculate_loss_of_path(send_t, recv_t):
	assert len(send_t) == len(recv_t)
	loss_rec = []
	loss_idx_plot = []
	ratios = []
	for i in range(len(send_t)):
		# print send_t[i]
		transformed_recv_t = [item[:-1] for item in recv_t[i]]
		temp_loss_idx = []
		temp_loss = list(set(send_t[i]) - set(transformed_recv_t))
		loss_rec.append(temp_loss)
		for loss in temp_loss:
			temp_loss_idx.append(send_t[i].index(loss))
		loss_idx_plot.append(temp_loss_idx)
		ratios.append(float(len(temp_loss))/len(send_t[i])*100)
	return loss_rec, loss_idx_plot, ratios

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
			next_group.append(traces[i][0][0])

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

def fec_decoding(trace, callback_pkts):
	pkt_need_callback = []
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
			if not isinstance(trace[recv_idx][0], int):
				curr_p_id = int(trace[recv_idx][0][0])
			else:
				curr_p_id = trace[recv_idx][0]
			if curr_p_id > e_id:
				# Enter the next group, jump out
				break
			recv_idx += 1

		curr_group = trace[pre_idx:recv_idx]
		# print curr_group
		group_idx += 1
		# Decoding
		if len(curr_group) >= ORI_SIZE:
			# Get the received pkts
			recved_group = trace[recv_idx-len(curr_group):recv_idx]
			recv_time = recved_group[ORI_SIZE-1][-1]	# The time of the last packet which makes decoding available
			decoded_time = recv_time + FEC_DE_TIME
			# Find callback pkts
			expected_id = range(s_id,e_id+1)
			for j in range(ORI_SIZE):
				if isinstance(recved_group[j][0], int):
					recved_group[j].append(decoded_time)
					assert recved_group[j][0] in expected_id
					expected_id.remove(recved_group[j][0])
					decoded_trace.append(recved_group[j])
				else:
					break
			#The number left in expected group is pkts need callback
			for cb in expected_id:
				assert cb == callback_pkts[cb][0]
				callback_pkts[cb].extend([-1, -1, decoded_time])
				decoded_trace.append(callback_pkts[cb])
		else:
			# decoded_trace.extend([x for x in curr_group if isinstance(x, int)])
			n_unrecoved += len([x for x in curr_group if isinstance(x[0], int)])
			print "Failed group:", range(s_id, e_id+1), ",", [x[0] for x in curr_group if isinstance(x[0], int)], "is received"
	# Process the remainning packets
	recv_packets_last = trace[recv_idx:]
	if len(recv_packets_last) >= n_left_pkts and n_left_pkts > 0:
		expected_id = range(recv_idx, SIM_LEN)
		decoded_time = recv_packets_last[n_left_pkts-1][-1] + FEC_DE_TIME
		for j in range(n_left_pkts):
			if isinstance(recv_packets_last[j][0], int):
				recv_packets_last[j].append(decoded_time)
				assert recv_packets_last[j][0] in expected_id
				expected_id.remove(recv_packets_last[j][0])
				decoded_trace.append(recv_packets_last[j])
			else:
				break
		for cb in expected_id:
			assert cb == callback_pkts[cb][0]
			callback_pkts[cb].extend([-1, -1, decoded_time])
			decoded_trace.append(callback_pkts[cb])
	else:
		n_unrecoved += len([x for x in recv_packets_last if isinstance(x, int)])
		# decoded_trace.extend([x for x in recv_packets_last if isinstance(x, int)])
	# print decoded_trace
	return decoded_trace, n_unrecoved

def generate_sending_encoding_pkts():
	curr_t = 0.0
	#Generate sending time gaps
	sending_time = np.random.exponential(PKT_GAP_IN_MS, SIM_LEN-1)
	sending_time = np.insert(sending_time, 0, 0.0)
	p_id = 0
	pre_encoding = []
	sending_pkts_trace = []
	total_n_fecs = 0
	while p_id < SIM_LEN:
		n_ori_pkts = np.minimum(ORI_SIZE, SIM_LEN - p_id)
		curr_group = []
		for i in range(n_ori_pkts):
			# Get the sending time of pkts
			curr_t += sending_time[p_id]
			curr_group.append([p_id, curr_t])
			pre_encoding.append([p_id, curr_t])		# Only for callback checking sending time
			p_id += 1
		# Then encoding
		encoded_time = curr_t + FEC_TIME		# FECT TIME could be a function of group size
		for pkt in curr_group:
			pkt.append(encoded_time)

		fec_title = str(curr_group[-1][0])
		for i in range(FEC_SIZE):
			curr_group.append([fec_title + '_fec_' + str(i), -1, encoded_time])
			total_n_fecs += 1
		sending_pkts_trace.extend(curr_group)
	return sending_pkts_trace, total_n_fecs, pre_encoding

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

def get_tp(trace):
	# Do later
	pass

def get_delay_info(trace):
	# print trace
	ave_delay = np.mean([pkt[4] - pkt[1] for pkt in trace])		# Decoded time minus sending time
	gap = np.mean([trace[i][4] - trace[i-1][4] for i in range(1, len(trace))])
	return ave_delay, gap

def main():
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

	# Intial sending time
	encoded_pkts, total_n_fecs, pre_encoding_pkts = generate_sending_encoding_pkts()
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

	call_back_pkts = pre_encoding_pkts[:]
	if IS_MULTIPATH:
		splited_traces = split_trace(encoded_pkts)
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
		rec_trace = single_path_deliver(encoded_pkts, cum_e_vec, cum_mat)

	decoded_trace, n_wasted_pkts = fec_decoding(rec_trace, call_back_pkts)
	print decoded_trace
	# Get delay here
	throughput = get_tp(decoded_trace)
	latency, jitter = get_delay_info(decoded_trace)


	print SIM_LEN, "source packets and", total_n_fecs, "FEC packets,", SIM_LEN + total_n_fecs, "packets in total" 
	print "Overhead ratio is: ", float(total_n_fecs)/SIM_LEN
	print len(rec_trace), "packets are received."
	# print len(decoded_trace), "source packets are recoved"
	print len(decoded_trace) + n_wasted_pkts, "source packets are received"
	if IS_MULTIPATH:
		for i in range(len(loss_rec)):
			print "Packets lost on path", i, "is", len(loss_rec[i]), ", loss ratio is", ratio_rec[i]
	print "Total loss ratio is:", (1.0 - (float(len(decoded_trace)) + n_wasted_pkts)/SIM_LEN) * 100,"%"
	print "Average delay:", latency
	print "Receiving Jitter:", jitter

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


if __name__ == '__main__':
	main()