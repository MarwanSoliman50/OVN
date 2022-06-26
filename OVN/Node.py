class Node(object):
    def __init__(self, node_dict):
        self._label = node_dict['label']
        self._position = node_dict['position']
        self._connected_nodes = node_dict['connected_nodes']
        self._successive = {}
        self._transceiver = ''

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, transceiver):
        self._transceiver = transceiver

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path

        if len(path) > 1:
            line_label = path[:2]  # Label to next Node for Line is created
            line = self.successive[line_label]  # Saves the NEXT node
            lightpath.next()  # Signal Path is Updated

            # Updates the signal_information propagating it,
            # then recursively sending it to the Line that then sends it to the
            # next node to collapse it all back in the end
            lightpath.signal_power = line.optimized_launch_power()
            lightpath = line.propagate(lightpath, occupation)

        return lightpath

    def probe(self, signal_information):
        path = signal_information.path

        if len(path) > 1:
            line_label = path[:2]  # Label to next Node for Line is created
            line = self.successive[line_label]  # Saves the NEXT node
            signal_information.next()  # Signal Path is Updated

            signal_information = line.probe(signal_information)

        return signal_information
