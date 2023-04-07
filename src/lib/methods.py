import numpy as np
import tqdm
import matplotlib.pylab as plt
from matplotlib import gridspec
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
        self.predict_ans = np.array(ans)
        if return_reservoir: 
            return self.predict_ans, self.reservoir_predict    
        else: 
            return self.predict_ans

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
        self.freerun_ans = np.array(ans)
        if return_reservoir: 
            return self.freerun_ans, self.reservoir_freerun       
        else: 
            return self.freerun_ans
    
    def computing_lyapunov_exponent(self, dt):
        W_dict = {"W_in": self.W_in, "W_res": self.W_res, "W_out": self.W_out}
        lyapunov_exponents, lyapunov_dim = lyapunov_exponent(self.reservoir_freerun, W_dict, self.freerun_length, dt)
        self.lyapunov_exponents = lyapunov_exponents
        self.lyapunov_dim = lyapunov_dim
        return self.lyapunov_exponents, self.lyapunov_dim
    

class Figure():
    def __init__(self, rcParams_dict):
        for key in rcParams_dict.keys():
            plt.rcParams[str(key)] = rcParams_dict[str(key)]

    def plt_timeseries_of_data_and_model(
                                        self,
                                        data,
                                        model,
                                        t,
                                        n_plot=None, 
                                        save_filename=None,
                                        params={'figsize':(20, 4),
                                                'linestyle_data':'-',
                                                'c_data':'k',
                                                'lw_data':3,
                                                'label_data':'Data',
                                                'linestyle_model':'--',
                                                'c_model':'r',
                                                'lw_model':3,
                                                'label_model':'Model',
                                                'legend':True,
                                                'legend_loc':'upper right',
                                                'xlabel':'Time [s]',
                                                'ylabel':r'$x(t)$'
                                                }
                                        ):
        fig = plt.figure(figsize=params['figsize'])
        ax = fig.add_subplot(111)
        ax.plot(t[:n_plot], data[:n_plot], linestyle=params['linestyle_data'], c=params['c_data'], lw=params['lw_data'], label=params['label_data'])
        ax.plot(t[:n_plot], model[:n_plot], linestyle=params['linestyle_model'], c=params['c_model'], lw=params['lw_model'], label=params['label_model'])
        if params['legend']:
            ax.legend(loc=params['legend_loc'])
        ax.set_xlabel(params['xlabel'])
        ax.set_ylabel(params['ylabel'])
        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename, bbox_inches="tight")
        
    def plt_attractor_of_data_and_model(
                                    self,
                                    data,
                                    model,
                                    n_plot=None, 
                                    n_shift=10,
                                    same_lim=True,
                                    transpose_model=False,
                                    save_filename=None,
                                    params={'figsize':(10, 4),
                                            'title_data':'Data',
                                            'linestyle_data':'-',
                                            'c_data':'k',
                                            'lw_data':2,
                                            'title_model':'Model',
                                            'linestyle_model':'-',
                                            'c_model':'r',
                                            'lw_model':2,
                                            'xlabel':r'$x(t)$',
                                            'ylabel':r'$x(t+\tau)$'
                                            }
                                    ):
        fig = plt.figure(figsize=params['figsize'])
        spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[1, 1],
                         wspace=0.6
                         )
        ax1 = fig.add_subplot(spec[0])
        ax1.set_title(params['title_data'], loc='center')
        ax1.plot(data[:-n_shift][:n_plot], data[n_shift:][:n_plot], linestyle=params['linestyle_data'], c=params['c_data'], lw=params['lw_data'])
        ax1.set_xlabel(params['xlabel'])
        ax1.set_ylabel(params['ylabel'])
        ax1.set_aspect('equal', 'datalim')
        ax2 = fig.add_subplot(spec[1])
        ax2.set_title(params['title_model'], loc='center')
        if transpose_model:
            ax2.plot(model[n_shift:][:n_plot], model[:-n_shift][:n_plot], linestyle=params['linestyle_model'], c=params['c_model'], lw=params['lw_model'])
        else:
            ax2.plot(model[:-n_shift][:n_plot], model[n_shift:][:n_plot], linestyle=params['linestyle_model'], c=params['c_model'], lw=params['lw_model'])
        ax2.set_xlabel(params['xlabel'])
        ax2.set_ylabel(params['ylabel'])
        if same_lim:
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(ax1.get_ylim())
        ax2.set_aspect('equal', 'datalim')
        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename, bbox_inches="tight")

    def plt_lyapunov_exponents(self,
                               lyapunov_exponents, 
                               n_dim=4,
                               lyapunov_lim=(-120, 50), 
                               save_filename=None,
                               params={
                                    'figsize':(8, 4),
                                    'linestyle_0line':'dashed',
                                    'c_0line':'b',
                                    'lw_0line':3,
                                    'linestyle_model':'-',
                                    'marker_model':'o',
                                    'markersize_model':10,
                                    'c_model':'r',
                                    'lw_model':3,
                                    'xlabel':'Lyapunov exponents',
                                    'ylabel':'Dimension'
                               }
                               ):
        fig = plt.figure(figsize=params['figsize'])
        ax = fig.add_subplot(111)
        ax.axhline(y=0, xmin=0, xmax=n_dim+1, linestyle=params['linestyle_0line'], c=params['c_0line'], lw=params['lw_0line'])
        ax.plot(np.arange(1, n_dim+1), lyapunov_exponents[:n_dim], linestyle=params['linestyle_model'], c=params['c_model'], 
                lw=params['lw_model'], marker=params['marker_model'], markersize=params['markersize_model'])
        ax.grid()
        ax.set_ylim(lyapunov_lim)
        ax.set_xlabel(params['xlabel'])
        ax.set_ylabel(params['ylabel'])
        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename, bbox_inches="tight")

    def plt_2attractors_powerspectra_lyapunov(self, data, model, t, lyapunov_exponents, 
                                              n_shift=10, n_initdel=2000, n_plt=None, same_lim=True, transpose_model=False,
                                              freq_lim=(50, 300),
                                              n_dim=4, lyapunov_lim=(-100, 70),
                                              save_filename=None):
        spec = gridspec.GridSpec(ncols=4, nrows=1, width_ratios=[4, 4, 7, 8], wspace=0.5)
        fig = plt.figure(figsize=(30, 4))
        freq_data, amp_data = fft(data, t)
        freq_model, amp_model = fft(model, t)

        ax0 = fig.add_subplot(spec[0])
        ax0.set_title('Data', loc='center')
        ax0.plot(data[n_initdel:][:-n_shift][:n_plt], data[n_initdel:][n_shift:][:n_plt], linestyle='-', c='k', lw=3)
        ax0.set_xlabel(r'$x(t)$')
        ax0.set_ylabel(r'$x(t+\tau)$')
        ax0.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax0.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax0.set_aspect('equal', 'datalim')

        ax1 = fig.add_subplot(spec[1])
        ax1.set_title('Model', loc='center')
        if transpose_model:
            ax1.plot(model[n_initdel:][n_shift:][:n_plt], model[n_initdel:][:-n_shift][:n_plt], linestyle='-', c='r', lw=3)
        else:
            ax1.plot(model[n_initdel:][:-n_shift][:n_plt], model[n_initdel:][n_shift:][:n_plt], linestyle='-', c='r', lw=3)
        if same_lim:
            ax1.set_xlim(ax0.get_xlim())
            ax1.set_ylim(ax0.get_ylim())
        ax1.set_xlabel(r'$x(t)$')
        ax1.set_ylabel(r'$x(t+\tau)$')
        ax1.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax1.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax1.set_aspect('equal', 'datalim')

        ax2 = fig.add_subplot(spec[2])
        ax2.set_title('Power Spectra', loc='center')
        ax2.plot(freq_data, amp_data, lw=4, c='k', label='Data')
        ax2.plot(freq_model, amp_model, '--', lw=4, c='r', label='Model')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Amplitude')
        ax2.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
        ax2.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax2.set_xlim(freq_lim)
        ax2.legend()

        ax3 = fig.add_subplot(spec[3])
        ax3.set_title('Lyapunov Exponent', loc='center')
        ax3.axhline(y=0, xmin=0, xmax=n_dim+1, linestyle='dashed', c='b', lw=3)
        ax3.plot(np.arange(1, n_dim+1), lyapunov_exponents[:n_dim], linestyle='-', c='r', 
                lw=3, marker='o', markersize=10)
        ax3.grid()
        ax3.set_ylim(lyapunov_lim)
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Lyapunov exponents')

        plt.tight_layout()
        if save_filename==None:
            plt.show()
        else:
            plt.savefig(save_filename, bbox_inches="tight")
