from scipy.constants import c, Planck
import numpy as np


class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._successive = {}
        self._state = ['free']*10
        self._gain = 16
        self._noise_figure = 3
        self._amplifiers = int(np.ceil(self._length / 80e3))
        self._span_length = self._length / self._amplifiers

        # Physical features
        self._adb = 0.2e-3  # dB/m (alpha dB)
        self._b2 = 2.13e-26  # (m Hz^2)^(-1) |beta2|
        self._gamma = 1.27e-3  # (m W)^(-1)
        self._Rs = 30e9
        self._df = 50e9

    @property
    def adb(self):
        return self._adb

    @property
    def b2(self):
        return self._b2

    @property
    def gamma(self):
        return self._gamma

    @property
    def Rs(self):
        return self._Rs

    @property
    def df(self):
        return self._df

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def amplifiers(self):
        return self._amplifiers

    @property
    def span_length(self):
        return self._span_length

    @property
    def label(self):
        return self._label

    @property
    def state(self):
        return self._state

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @state.setter
    def state(self,state):
        state = [s.lower().strip() for s in state]
        if set(state).issubset(['free', 'occupied']):
            self._state = state
        else:
            print('ERROR: line state not recognized. Value:', set(state) - set(['free', 'occupied']))

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def optimized_lauch_power(self, signal_power):
        ase = self.ase_generation()
        eta = self.nli_generation(signal_power)
        p_opt = (ase / (2 * eta)) ** (1 / 3)  # Calculate optimum signal power

    def nli_generation(self, signal_power, Rs, df): # Nonlinear Interface Noise
        Bn = 12.5e9  # noise bandwidth
        N_spans = self.amplifiers
        nnli = 0  #come si fa???
        nil = signal_power**3 * nnli * N_spans * Bn

        return nil

    def ase_generation(self): # Amplified Spontaneous Emissions
        gain_lin = 10 ** (self._gain / 10)
        noise_figure_lin = 10 ** (self._noise_figure / 10)
        N = self.amplifiers
        f = 193.414e12
        h = Planck
        Bn = 12.5e9
        ase_noise = N * h * f * Bn * noise_figure_lin * (gain_lin - 1)
        return ase_noise

    def latency_generation(self):
        latency = self.length / (c * 2 / 3)
        return latency

    def noise_generation(self, lightpath):
        # noise = signal_power / (2 * self.length)
        noise = self.ase_generation() + self.nli_generation(lightpath.signal_power, not self.Rs, self.df)
        return noise

    def propagate(self, lightpath, occupation=False):
        # Update latency
        latency = self.latency_generation()
        lightpath.add_latency(latency)

        # Update noise
        noise = self.noise_generation(lightpath)
        lightpath.add_noise(noise)

        # Update line state
        if occupation:
            channel = lightpath.channel
            new_state = self.state.copy()
            new_state[channel] = 'occupied'
            self.state = new_state

        node = self.successive[lightpath.path[0]]  # Finds next Node
        lightpath = node.propagate(lightpath, occupation)  # Sends the updated Signal to the next Node
        return lightpath  # Returns the Signal that has previously been recursively send forward

    def probe(self, signal_information):
        # Update latency
        latency = self.latency_generation()
        signal_information.add_latency(latency)

        # Update noise
        #signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_information)
        signal_information.add_noise(noise)

        node = self.successive[signal_information.path[0]]  # Finds next Node
        signal_information = node.probe(signal_information)  # Sends the updated Signal to the next Node
        return signal_information  # Returns the Signal that has previously been recursively send forward
