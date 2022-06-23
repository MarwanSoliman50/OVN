from OVN import *
import matplotlib.pyplot as plt
from random import shuffle
import copy
import numpy as np
network = Network('/Users/marwan5050/Downloads/UNIVERSITA  STUDIO /3rd year /2nd/open & virtulaized networks/Codes/nodes.json')

network_fixed_rate=Network('/Users/marwan5050/Downloads/UNIVERSITA  STUDIO /3rd year /2nd/open & virtulaized networks/Codes/nodes_full_fixed_rate.json')
network_flex_rate=Network('/Users/marwan5050/Downloads/UNIVERSITA  STUDIO /3rd year /2nd/open & virtulaized networks/Codes/nodes_full_flex_rate.json')
network_shannon=Network('/Users/marwan5050/Downloads/UNIVERSITA  STUDIO /3rd year /2nd/open & virtulaized networks/Codes/nodes_full_shannon.json')


node_labels = list(network.nodes.keys())
connections = []
for i in range (100):
    shuffle(node_labels)
    connection =Connection(node_labels[0],node_labels[-1],1e-3)
    connections.append(connection)

connections1 = copy.deepcopy(connections)
connections2 = copy.deepcopy(connections)
connections3 = copy.deepcopy(connections)
bins = np.linspace(90, 700, 20)

streamed_connections_fixed = network.stream(connections1 ,best='snr')



snrs1=[connection.snr for connection in streamed_connections_fixed]
snr_s1=np.ma.masked_equal(snrs1,0)

plt.hist(snr_s1,bins=20)
plt.title('SNR Dist. full fixed_rate')
plt.xlabel('dB')
plt.show()

bit_rate_fixed=[connection.bit_rate for connection in streamed_connections_fixed]
bit_r=np.ma.masked_equal(bit_rate_fixed,0)
plt.hist(bit_r,bins,label='fixed_rate')
plt.title('Bit_Rate  full fixed_rate')
plt.xlabel('Gbps')
plt.show()



streamed_connections_flex = network.stream(connections2 ,best='snr')
snrs2=[connection.snr for connection in streamed_connections_flex]
snr_s2=np.ma.masked_equal(snrs2,0)

plt.hist(snr_s2,bins=20)
plt.title('SNR Dist. full flex_rate')
plt.xlabel('dB')
plt.show()

bit_rate_flex=[connection.bit_rate for connection in streamed_connections_flex]
bit_r2=np.ma.masked_equal(bit_rate_flex,0)
plt.hist(bit_r2,bins,label='flex_rate')
plt.title('Bit_Rate  full flex_rate')
plt.xlabel('Gbps')
plt.show()

streamed_connections_shannon = network.stream(connections3 ,best='snr')
snrs3=[connection.snr for connection in streamed_connections_shannon]
snr_s3=np.ma.masked_equal(snrs3,0)

plt.hist(snr_s3,bins=20)
plt.title('SNR Dist. full shannon_rate')
plt.xlabel('dB')
plt.show()

bit_rate_shannon=[connection.bit_rate for connection in streamed_connections_shannon]
bit_r3=np.ma.masked_equal(bit_rate_shannon,0)
plt.hist(bit_r3,bins,label='fixed_rate')
plt.title('Bit_Rate  full fixed_rate')
plt.xlabel('Gbps')
plt.show()



streamed_connections=network.stream(connections)
latencies=[connection.latency for connection in streamed_connections_shannon]
plt.hist(np.ma.masked_equal(latencies, 0), bins=25)
plt.title('Latency Distribution')
plt.savefig('Plots/LatencyDistribution.png')
plt.show()
snrs = [connection.snr for connection in streamed_connections_shannon]
plt.hist(np.ma.masked_equal(snrs, 0), bins=20)
plt.title('SNR Dstribution')
plt.savefig('Plots/SNRDistribution.png')
plt.show()
