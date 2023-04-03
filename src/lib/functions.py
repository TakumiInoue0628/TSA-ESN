import os
import random
import numpy as np
import tqdm
from scipy.linalg import qr
import pandas as pd
from scipy.signal import buttord, butter, filtfilt
from sklearn.metrics import mean_squared_error


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #tf.random.set_seed(seed)

def W_res_create(shape, SR=0.99, density=0.1, seed=0):
    set_seed(seed)
    W1 = np.random.uniform(size=shape, low=-1.0, high=1.0)
    W2 = np.random.uniform(size=shape, low=0.0, high=1.0)
    W2 = (W2 > (1.0 - density)).astype(np.float)
    W = W1 * W2
    value, _ = np.linalg.eigh(W)
    sr = max(np.abs(value))
    return W * SR / sr

def lyapunov_exponent(reservoir, W_dict, length, dt):
        ones = np.ones((W_dict["W_res"].shape[0], W_dict["W_res"].shape[1]))
        lyapunov_exponent = np.zeros((reservoir.shape[2]))
        Q = np.eye((W_dict["W_res"].shape[0]))
        W = W_dict["W_res"] + np.matmul(W_dict["W_out"], W_dict["W_in"])
        for i in tqdm.tqdm(range(length), desc="Computing Lyapunov Exponent", leave=False):
            S = reservoir[i,:,:].reshape(-1,1) ** 2
            J = (ones - S) * W.T
            Q, R = qr(np.matmul(J, Q))
            lmd = np.log(np.abs(np.diag(R)))
            lyapunov_exponent += lmd
        lyapunov_exponent = lyapunov_exponent / (length*dt)
        l = 0
        for i in range(length):
            l += lyapunov_exponent[i]
            if l < 0: break
        dim = i + (sum(lyapunov_exponent[:i]) / abs(lyapunov_exponent[i]))
        if dim <= 1.0: dim = 1.0
        return lyapunov_exponent, dim

def load_csv_data(csv_path, data_name_list, sample_span):
    print('Loading csv data')
    print('file path | '+csv_path)
    print('data list | '+", ".join(data_name_list))
    data_df = pd.read_csv(csv_path)
    data_list = []
    for i in range(len(data_name_list)):
        data_list.append(data_df[[data_name_list[i]]].values[sample_span[0]:sample_span[1], 0])
    index = np.arange(sample_span[0], sample_span[1])
    return data_list, index

def bandpass_filter(data, t_data, passband_edge_freq, stopband_edge_freq, passband_edge_max_loss, stopband_edge_min_loss,):
    dt = t_data[1] - t_data[0]
    sampling_rate = 1. / dt
    niquist_freq = sampling_rate / 2.
    passband_edge_freq_normalize = passband_edge_freq / niquist_freq
    stopband_edge_freq_normalize = stopband_edge_freq / niquist_freq
    butterworth_order, butterworth_natural_freq = buttord(
                                                        wp=passband_edge_freq_normalize, 
                                                        ws=stopband_edge_freq_normalize,
                                                        gpass=passband_edge_max_loss,
                                                        gstop=stopband_edge_min_loss
                                                        )
    numerator_filterfunc, denominator_filterfunc = butter(
                                                        N=butterworth_order,
                                                        Wn=butterworth_natural_freq,
                                                        btype='band'
                                                        )
    data_filtered = filtfilt(
                            b=numerator_filterfunc,
                            a=denominator_filterfunc,
                            x=data
                            )
    return data_filtered

def rmse(y_true, y_pred):
    score = []
    for i in range(y_true.shape[0]):
        score.append(mean_squared_error(y_true[i,:], y_pred[i,:], squared=False))
    return np.array(score)