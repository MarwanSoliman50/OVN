import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .Line import Line
from .Signalinformation import SignalInformation, Lightpath
from .Node import Node
from .Connection import Connection
from scipy import special as sp
from random import shuffle
from datetime import datetime


BER_t = 1e-3
Bn = 12.5e9  # noise bandwidth
Rs = 32.0e9  # symbol-rate of light path


class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path, 'r'))  # Load Json nodes file
        self._nodes = {}
        self._lines = {}
        self._weighted_paths = None
        self._connected = False
        self._route_space = None
        self._traffic_matrix = 0
        self._suc_connections = 0
        self._logger = pd.DataFrame(columns=['epoch_time', 'path', 'channel_ID', 'br'])  # pd.DataFrame(columns=['epoch_time', 'path', 'channel_ID', 'br'])

        for node_label in node_json:
            # Create the node instance
            node_dict = node_json[node_label]  # Saves the node dictionary in node_dict
            # for ex: {'connected_nodes': ['B', 'C', 'D'], 'position': [-350000.0, 150000.0], 'label': 'A'} for node A

            node_dict['label'] = node_label  # Looks useless the label is already there ?????????
            node = Node(node_dict)  # Creates the object Node for the single nodes
            if 'transceiver' not in node_dict.keys():
                node.transceiver = 'fixed_rate'
            else:
                node.transceiver = node_dict['transceiver']
            self._nodes[node_label] = node  # Adds the node Object to the dictionary

            # Create the line instances
            for connected_node_label in node_dict['connected_nodes']:  # Iterates through all connected nodes
                line_dict = {}  # Temp dict to then add to the main dict
                line_label = node_label + connected_node_label  # Creates the label for the line
                line_dict['label'] = line_label  # Adds the label as an attribute
                node_position = np.array(node_json[node_label]['position'])  # Gets node position
                connected_node_position = np.array(
                    node_json[connected_node_label]['position'])  # Gets connected node position

                # Calculates distance of two points
                line_dict['length'] = np.sqrt(np.sum((node_position - connected_node_position) ** 2))
                line = Line(line_dict)  # Creates the line object
                self._lines[line_label] = line  # Adds the object to the dictionary

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def suc_connections(self):
        return self._suc_connections

    @property
    def traffic_matrix(self):
        return self._traffic_matrix

    @traffic_matrix.setter
    def traffic_matrix(self, traffic_matrix):
        self._traffic_matrix = traffic_matrix

    @property
    def route_space(self):
        return self._route_space

    @property
    def connected(self):
        return self._connected

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    def draw(self):
        nodes = self.nodes
        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0] / 1e3
            y0 = n0.position[1] / 1e3
            plt.plot(x0, y0, 'go', markersize=10)
            plt.text(x0 + 30, y0 + 30, node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0] / 1e3
                y1 = n1.position[1] / 1e3
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('Network')
        plt.show()

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        start_node = self.nodes[path[0]]
        propagated_lightpath = start_node.propagate(lightpath, occupation)
        return propagated_lightpath

    def probe(self, signal_information):
        path = signal_information.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.probe(signal_information)
        return propagated_signal_information

    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines

        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]
        self._connected = True

    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys() if ((key != label1) & (key != label2))]
        cross_lines = self.lines.keys()
        inner_paths = {}
        inner_paths['0'] = label1
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [inner_path + cross_node
                                            for cross_node in cross_nodes
                                            if ((inner_path[-1] + cross_node in cross_lines) &
                                                (cross_node not in inner_path))]
        paths = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)
        return paths

    def available_paths(self, input_node, output_node):
        if self._weighted_paths is None:
            self.weighted_paths(1)

        all_paths = [path for path in self.weighted_paths.path.values
                     if ((path[0] == input_node) and (path[-1] == output_node))]
        available_paths = []
        for path in all_paths:
            path1 = path.replace('->', '')
            path_occupancy = self.route_space.loc[
                self.route_space.path == path1].T.values

            path_occupancy = [x for l in path_occupancy for x in l]

            flag = True
            for i in path_occupancy:
                if i == 'free':
                    flag = False

            if not flag:
                available_paths.append(path)
        return available_paths

    def set_weighted_paths(self, signal_power):
        if not self.connected:
            self.connect()
        node_labels = self.nodes.keys()
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label1 != label2: pairs.append(label1 + label2)
        # columns = ['path', 'latency', 'noise', 'snr']
        df = pd.DataFrame()
        paths = []
        latencies = []
        noises = []
        snrs = []
        for pair in pairs:
            for path in self.find_paths(pair[0], pair[1]):
                path_string = ''
                for node in path:
                    path_string += node + '->'
                paths.append(path_string[:-2])
                # Propagation
                signal_information = SignalInformation(signal_power, path)
                signal_information = self.probe(signal_information)
                latencies.append(signal_information.latency)

                noises.append(signal_information.noise_power)
                snrs.append(10 * np.log10(signal_information.signal_power / signal_information.noise_power))

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        self._weighted_paths = df

        paths_rs = [i.replace('->', '') for i in paths]
        route_space = pd.DataFrame()
        route_space['path'] = paths_rs
        for i in range(10):
            route_space[str(i)] = ['free'] * len(paths)
            self._route_space = route_space

    def find_best_snr(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_snr = np.max(inout_df.snr.values)
            best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0].replace('->', '')
            return best_path
        else:
            best_path = None
            return best_path

    def find_best_latency(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_latency = np.min(inout_df.latency.values)
            best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0].replace('->', '')
            return best_path
        else:
            best_paths = None
            return best_paths

    def stream(self, connections, best='latency'):
        streamed_connections = []
        for connection in connections:
            input_node = connection.input_node
            output_node = connection.output_node
            signal_power = connection.signal_power
            self.set_weighted_paths(signal_power)
            if best == 'latency':
                path = self.find_best_latency(input_node, output_node)
            elif best == 'snr':
                path = self.find_best_snr(input_node, output_node)
            else:
                print('ERROR: best input not recognized.Value:', best)
                continue
            flag = True

            for i in range(len(path)-1):
                if not self.lines[path[i]+path[i+1]].in_service:
                    flag = False

            if path and flag:  # in case path is not None

                path_occupancy = self.route_space.loc[
                    self.route_space.path == path].T.values
                channel = [i for i in range(len(path_occupancy))
                           if path_occupancy[i] == 'free'][0]

                lightpath = Lightpath(signal_power, path, channel)
                rb = self.calculate_bit_rate(lightpath, self.nodes[input_node].transceiver)
                connection.bit_rate = rb

                if rb != 0:
                    in_signal_information = Lightpath(signal_power, path, channel)
                    out_signal_information = self.propagate(in_signal_information)
                    connection.latency = out_signal_information.latency
                    noise = out_signal_information.noise_power
                    connection.snr = 10 * np.log10(signal_power / noise)
                    self.update_route_space(path, channel, rb)
                    streamed_connections.append(connection)

            else:
                connection.latency = None
                connection.snr = 0
                streamed_connections.append(connection)
        return streamed_connections

    @staticmethod
    def path_to_line_set(path):
        path = path.replace('->', '')
        return set([path[i] + path[i + 1] for i in range(len(path) - 1)])

    @staticmethod
    def line_to_path_set(path):
        sol = []
        for i in range(len(path) -1):
            sol.append(path[i])
            sol.append('->')
        sol.append(path[-1])
        path = ''.join(sol)
        return path

    def update_route_space(self, path, channel, rb):
        all_paths = [self.path_to_line_set(p)
                     for p in self.route_space.path.values]

        states = self.route_space[str(channel)]
        lines = self.path_to_line_set(path)
        for i in range(len(all_paths)):
            line_set = all_paths[i]
            if lines.intersection(line_set):
                states[i] = 'occupied'

        self.route_space[str(channel)] = states
        self.update_logger([datetime.timestamp(datetime.now()), path, channel, rb])

    def calculate_bit_rate(self, lightpath, strategy):
        global BER_t
        Rs = lightpath.Rs
        global Bn
        rb = 0
        path = self.line_to_path_set(lightpath.path)
        GSNR_db = pd.array(self.weighted_paths.loc[self.weighted_paths['path'] == path]['snr'])[0]
        GSNR = 10 ** (-GSNR_db / 10)

        if strategy == 'fixed_rate':
            if GSNR >= 2 * sp.erfcinv(2 * BER_t) ** 2 * (Rs / Bn):
                rb = 100
            else:
                rb = 0

        if strategy == 'flex_rate':
            if GSNR < 2 * sp.erfcinv(2 * BER_t) ** 2 * (Rs / Bn):
                rb = 0
            elif (GSNR >= 2 * sp.erfcinv(2 * BER_t) ** 2 * (Rs / Bn)) & (GSNR < (14 / 3) * sp.erfcinv(
                    (3 / 2) * BER_t) ** 2 * (Rs / Bn)):
                rb = 100
            elif (GSNR >= (14 / 3) * sp.erfcinv((3 / 2) * BER_t) ** 2 * (Rs / Bn)) & (GSNR < 10 * sp.erfcinv(
                    (8 / 3) * BER_t) ** 2 * (Rs / Bn)):
                rb = 200
            elif GSNR >= 10 * sp.erfcinv((8 / 3) * BER_t) ** 2 * (Rs / Bn):
                rb = 400

        if strategy == 'shannon':
            rb = 2 * Rs * np.log2(1 + Rs * GSNR / Bn) / 1e9

        return rb

    def stream2(self, connections, best='latency'):
        streamed_connections = []
        for connection in connections:
            input_node = connection.input_node
            output_node = connection.output_node
            signal_power = connection.signal_power
            self.set_weighted_paths(signal_power)
            if best == 'latency':
                path = self.find_best_latency(input_node, output_node)
            elif best == 'snr':
                path = self.find_best_snr(input_node, output_node)
            else:
                print('ERROR: best input not recognized.Value:', best)
                continue
            if path:  # in case path is not None

                path_occupancy = self.route_space.loc[
                    self.route_space.path == path].T.values
                channel = [i for i in range(len(path_occupancy))
                           if path_occupancy[i] == 'free'][0]

                lightpath = Lightpath(signal_power, path, channel)
                rb = self.calculate_bit_rate(lightpath, self.nodes[input_node].transceiver)
                connection.bit_rate = rb

                if rb != 0:
                    in_signal_information = Lightpath(signal_power, path, channel)
                    out_signal_information = self.propagate(in_signal_information)
                    connection.latency = out_signal_information.latency
                    noise = out_signal_information.noise_power
                    connection.snr = 10 * np.log10(signal_power / noise)
                    self.update_route_space(path, channel, signal_power)
                    streamed_connections.append(connection)

                    self.allowed_connection(input_node,output_node,rb) # Matrix calculation

            else:
                connection.latency = None
                connection.snr = 0
                streamed_connections.append(connection)
        return streamed_connections

    def create_traffic_matrix(self, multiplier=1):
        s = pd.Series(data=[0.0] * len(self.nodes), index=self.nodes.keys())
        df = pd.DataFrame(float(100 * multiplier), index=s.index, columns=s.index, dtype=s.dtype)
        np.fill_diagonal(df.values, s)
        self._suc_connections = 0
        return df

    def matrix_calc(self, connections, node_labels):
        su_connections = []
        for j in range(15):
            self.traffic_matrix = self.create_traffic_matrix(j + 1)
            for i in range(100):
                shuffle(node_labels)
                connection = Connection(node_labels[0], node_labels[-1], 1)
                connections.append(connection)

            self.stream(connections, best='snr')
            print(j)
            su_connections.append(self.suc_connections)
        print(su_connections)

    def allowed_connection(self, input_node, output_node, rb):
        if rb <= self.traffic_matrix[input_node][output_node]:
            self.traffic_matrix[input_node][output_node] -= rb
            self._suc_connections += 1

    def update_logger(self, data):
        df = pd.DataFrame([data], columns=['epoch_time', 'path', 'channel_ID', 'br'])
        self.logger = pd.concat([self.logger, df], ignore_index=True)

    def strong_failure(self, label):
        self.lines[label].in_service = 0

