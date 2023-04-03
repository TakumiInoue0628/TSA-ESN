import numpy as np
import tqdm
import matplotlib.pylab as plt
from lib.functions import *


def LoadCSV(file_path, data_name_list, sample_span):
    data_list, _ = load_csv_data(file_path, data_name_list, sample_span)
    return data_list


class PreProcessing():

    def __init__(self, data, t_data):
        self.raw_data = data
        self.data = data
        self.t_data = t_data

    def cut(self, span, new_t=False):
        self.data = self.data[span[0]:span[1]]
        self.t_data = self.t_data[span[0]:span[1]]
        if new_t: ### t0=0, t1=0+dt, t2=0+2dt ...
            self.t_data = np.arange(0, self.t_data.shape[0])*(self.t_data[1]-self.t_data[0])

    def filter(self, method='bandpass_filtering', params={'passband_edge_freq':np.array([90, 200]), 'stopband_edge_freq':np.array([20, 450]), 'passband_edge_max_loss':1, 'stopband_edge_min_loss':10}):
        if method=='bandpass_filtering':
            self.data = bandpass_filter(
                                        data=self.data,
                                        t_data=self.t_data,
                                        passband_edge_freq=params['passband_edge_freq'],
                                        stopband_edge_freq=params['stopband_edge_freq'],
                                        passband_edge_max_loss=params['passband_edge_max_loss'],
                                        stopband_edge_min_loss=params['stopband_edge_min_loss'],
                                        )
        else:
            print('There is no such method.')

    def embed(self, n_shift, n_dimension):
        length = len(self.data) - ((n_dimension-1)*n_shift)
        data_embedded = np.zeros((length, n_dimension))
        for i in range(n_dimension):
            data_embedded[:, i] = np.roll(self.data, -i*n_shift)[:-((n_dimension-1)*n_shift)]
        self.data_embedded = data_embedded
        self.t_data_embedded = self.t_data[:data_embedded.shape[0]]
        self.n_shift = n_shift
        self.n_dimension = n_dimension

    def train_test_split(self, n_train, n_predstep):
        train_X = self.data_embedded[:(n_train-n_predstep-(self.data_embedded.shape[1]-1)*self.n_shift)].reshape(-1, 1, self.data_embedded.shape[1])
        train_Y = self.data_embedded[n_predstep:(n_train-(self.data_embedded.shape[1]-1)*self.n_shift)].reshape(-1, 1, self.data_embedded.shape[1])
        test_X = self.data_embedded[n_train:-n_predstep].reshape(-1, 1, self.data_embedded.shape[1])
        test_Y = self.data_embedded[n_train+n_predstep:].reshape(-1, 1, self.data_embedded.shape[1])
        return train_X, train_Y, test_X, test_Y


class EchoStateNetwork():

    def __init__(self, units=256, SR=0.99, input_shape=1, output_shape=1, W_in_scale=1.0, W_res_density=0.1, leak_rate=1.0, alpha=1.0e-4, seed=0):
        self.units = units
        self.SR = SR
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.W_in_scale = W_in_scale
        self.W_res_density = W_res_density
        self.leak_rate = leak_rate
        self.alpha = alpha
        set_seed(seed)
        self.W_in = np.random.uniform(size=(self.input_shape, self.units), low=-self.W_in_scale, high=self.W_in_scale)
        self.W_res = W_res_create(shape=(self.units, self.units), SR=self.SR, density=self.W_res_density)
        self.bias = np.random.uniform(size=(1, self.units), low=-0.1, high=0.1)
        self.x_n = np.random.uniform(size=(1, self.units))
        self.W_out =np.random.uniform(size=(self.units, self.output_shape))

    def fit(self, in_layer_data, out_layer_data):
        x = in_layer_data
        y = out_layer_data
        Xt_X, Xt_Y = 0.0, 0.0
        for i in tqdm.tqdm(range(x.shape[0]), desc="Learning", leave=False):
            In = np.matmul([x[i,:]], self.W_in)
            Res = np.matmul(self.x_n, self.W_res)
            self.x_n = ((1.0 - self.leak_rate) * self.x_n + self.leak_rate * np.tanh(In + Res)).reshape(1, self.units)
            y_n = (y[i,:]).reshape(1, self.output_shape)
            Xt_X += np.matmul(np.transpose(self.x_n), self.x_n)
            Xt_Y += np.matmul(np.transpose(self.x_n), y_n)
        Xt_X_aI = Xt_X + self.alpha * np.eye(int(self.units))
        self.W_out = np.matmul(np.linalg.inv(Xt_X_aI), Xt_Y)
        self.opt_x_n = self.x_n

    def predict(self, in_layer_data, return_reservoir=False):
        x = in_layer_data
        ans, reservoir = [], []
        for i in tqdm.tqdm(range(x.shape[0]), desc="Predicting (One Step)", leave=False):
            In = np.matmul([x[i,:]], self.W_in)
            Res = np.matmul(self.x_n, self.W_res)
            self.x_n = ((1.0 - self.leak_rate) * self.x_n + self.leak_rate * np.tanh(In + Res)).reshape(1, self.units)
            pred = np.matmul(self.x_n, self.W_out)
            ans.append(pred.reshape(-1).tolist())
            reservoir.append(self.x_n)
        self.reservoir_predict = np.array(reservoir)  
        if return_reservoir: 
            return np.array(ans), np.array(reservoir)       
        else: 
            return np.array(ans)

    def freerun(self, in_layer_data0, pred_range=100, return_reservoir=False):
        self.freerun_length = pred_range
        x = in_layer_data0
        ans, reservoir = [], []
        for _ in tqdm.tqdm(range(pred_range), desc="Predicting (Freerun)", leave=False):
            In = np.matmul([x], self.W_in)
            Res = np.matmul(self.x_n, self.W_res)
            self.x_n = ((1.0 - self.leak_rate) * self.x_n + self.leak_rate * np.tanh(In + Res)).reshape(1, self.units)
            pred = np.matmul(self.x_n, self.W_out)
            ans.append(pred.reshape(-1).tolist())
            reservoir.append(self.x_n)
            x = pred
        self.reservoir_freerun = np.array(reservoir)  
        if return_reservoir: 
            return np.array(ans), np.array(reservoir)       
        else: 
            return np.array(ans)
    
    def computing_lyapunov_exponent(self, dt):
        W_dict = {"W_in": self.W_in, "W_res": self.W_res, "W_out": self.W_out}
        return lyapunov_exponent(self.reservoir_freerun, W_dict, self.freerun_length, dt)
    

class Figure():
    def __init__(self, rcParams_dict):
        for key in rcParams_dict.keys():
            plt.rcParams[str(key)] = rcParams_dict[str(key)]