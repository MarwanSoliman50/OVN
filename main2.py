from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
from OVN import Network, Connection

network_flexed = Network('/Users/marwan5050/Downloads/UNIVERSITA  STUDIO /3rd year /2nd/open & virtulaized networks/Codes/OVN/271036_flex.json')
network_fixed = Network('/Users/marwan5050/Downloads/UNIVERSITA  STUDIO /3rd year /2nd/open & virtulaized networks/Codes/OVN/271036_fixed.json')
network_shannon = Network('/Users/marwan5050/Downloads/UNIVERSITA  STUDIO /3rd year /2nd/open & virtulaized networks/Codes/OVN/271036_shannon.json')

network_fixed.connect()
network_flexed.connect()
network_shannon.connect()

#network_fixed.draw()

node_labels_fixed = list(network_fixed.nodes.keys())
node_labels_flexed = list(network_flexed.nodes.keys())
node_labels_shannon = list(network_shannon.nodes.keys())
connections = []
# --------------------------------------------------------------------------------------------------
su_connections_flex = []
su_connections_fixed = []
su_connections_shannon = []

for i in range(100):
    shuffle(node_labels_fixed)
    connection = Connection(node_labels_fixed[0], node_labels_fixed[-1], 1)
    connections.append(connection)

for j in range(100):
    network_fixed.traffic_matrix = network_fixed.create_traffic_matrix(j + 1)
    network_fixed.stream2(connections, best='snr')
    print(j)
    su_connections_fixed.append(network_fixed.suc_connections)
    if network_fixed.suc_connections == 100:
        break

for j in range(100):
    network_flexed.traffic_matrix = network_flexed.create_traffic_matrix(j + 1)
    network_flexed.stream2(connections, best='snr')
    print(j)
    su_connections_flex.append(network_flexed.suc_connections)
    if network_flexed.suc_connections == 100:
        break

for j in range(100):
    network_shannon.traffic_matrix = network_shannon.create_traffic_matrix(j + 1)
    network_shannon.stream2(connections, best='snr')
    print(j)
    su_connections_shannon.append(network_shannon.suc_connections)
    if network_shannon.suc_connections == 100:
        break


plt.plot(su_connections_fixed)
plt.title('Saturation Rate Fixed')
plt.xlabel('M')
plt.ylabel('number of saturated requests')
plt.show()

plt.plot(su_connections_flex)
plt.title('Saturation Rate Flexed')
plt.xlabel('M')
plt.ylabel('number of saturated requests')
plt.show()

plt.plot(su_connections_shannon)
plt.title('Saturation Rate Shannon')
plt.xlabel('M')
plt.ylabel('number of saturated requests')
plt.show()

plt.plot(su_connections_flex)
plt.plot(su_connections_fixed)
plt.plot(su_connections_shannon)
plt.legend(('flexed', 'fixed', 'shannon'),
           loc='lower right')
plt.title('Saturation Rate Shannon')
plt.xlabel('M')
plt.ylabel('number of saturated requests')
plt.show()

# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
#
# for i in range(100):
#     shuffle(node_labels_fixed)
#     connection = Connection(node_labels_fixed[0],node_labels_fixed[-1],1)
#     connections.append(connection)
#
# streamed_connections_fixed = network_fixed.stream(connections, best='snr')
# bit_rate_fixed_rate = [connection.bit_rate for connection in streamed_connections_fixed]
#
# streamed_connections_flexed = network_flexed.stream(connections, best='snr')
# bit_rate_flexed_rate = [connection.bit_rate for connection in streamed_connections_flexed]
#
# streamed_connections_shannon = network_shannon.stream(connections, best='snr')
# bit_rate_shannon_rate = [connection.bit_rate for connection in streamed_connections_shannon]
#
# bins = np.linspace(0, 1500, 100)
#
# plt.hist(bit_rate_fixed_rate)
# plt.hist(bit_rate_flexed_rate)
# plt.hist(bit_rate_shannon_rate)
# plt.legend(('flexed', 'fixed', 'shannon'), loc='lower right')
# plt.title('BitRate all rates')
# plt.xlabel('Gbps')
# plt.show()

# --------------------------------------------------------------------------------------------------

# for i in range(100):
#     shuffle(node_labels_fixed)
#     connection = Connection(node_labels_fixed[0],node_labels_fixed[-1],1)
#     connections.append(connection)
#
# network_fixed.draw()

# Signal to Noise Ratio
streamed_connections = network_fixed.stream(connections, best='snr')

snrs = [connection.snr for connection in streamed_connections]
plt.hist(snrs, bins=20)

tot = sum(snrs)
avg = tot / len(snrs)
print(f"snrs average is: {avg}")

plt.title('SNR Distribution')
plt.xlabel('dB')
plt.show()

# Latency dist
streamed_connections = network_fixed.stream(connections, best='snr')

latencies = [connection.latency for connection in streamed_connections]
plt.hist(np.ma.masked_equal(latencies, 0), bins=25)

tot = sum(latencies)
avg = tot / len(latencies)
print(f"Latency average is: {avg}")

plt.title('Latency Distribution')
plt.xlabel('s')
plt.show()


# Full Fixed

#network.draw()
streamed_connections = network_fixed.stream(connections, best='snr')

bit_rate_fixed_rate = [connection.bit_rate for connection in streamed_connections]

tot = sum(bit_rate_fixed_rate)
print(f"Fixed tot capacity is: {tot}")

plt.hist(bit_rate_fixed_rate, label='fixed-rate')
plt.title('BitRate Full fixed-rate')
plt.xlabel('Gbps')
plt.show()

# # Full Flex
node_labels_flex = list(network_flexed.nodes.keys())
connections = []
for i in range(100):
    shuffle(node_labels_flex)
    connection = Connection(node_labels_flex[0],node_labels_flex[-1],1)
    connections.append(connection)

streamed_connections = network_flexed.stream(connections, best='snr')  # best='snr'

bit_rate_flexed_rate = [connection.bit_rate for connection in streamed_connections]

tot = sum(bit_rate_flexed_rate)
print(f"Flex tot capacity is: {tot}")

plt.hist(bit_rate_flexed_rate, label='flex-rate')
plt.title('BitRate Full flex-rate')
plt.xlabel('Gbps')
plt.show()
#
#
# Full Shannon
# node_labels_shannon = list(network_shannon.nodes.keys())
# connections = []
for i in range(100):
    shuffle(node_labels_shannon)
    connection = Connection(node_labels_shannon[0],node_labels_shannon[-1],1)
    connections.append(connection)

streamed_connections = network_shannon.stream(connections, best='snr')  # , best='snr'

bit_rate_shannon_rate = [connection.bit_rate for connection in streamed_connections]
tot = sum(bit_rate_shannon_rate)
avg = tot / len(bit_rate_shannon_rate)
print(f"Shannon tot capacity is: {tot}")
print(f"Shannon average is: {avg}")

plt.hist(bit_rate_shannon_rate, label='shannon-rate')
plt.title('BitRate Full shannon-rate')
plt.xlabel('Gbps')
plt.show()