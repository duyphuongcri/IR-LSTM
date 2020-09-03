import torch
import math
import numpy as np
import torch.nn 
import matplotlib.pyplot as plt
import torch.nn as nn

class Linear_diagonal_weight(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear_diagonal_weight,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))
        self.diagonal_mask = torch.eye(output_size, input_size).cuda()
    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight * self.diagonal_mask, self.bias)

class Linear_diagonal_weight_z(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear_diagonal_weight_z,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))
        self.diagonal_mask = torch.eye(output_size, input_size).cuda()
        self.diagonal_mask = 1 - self.diagonal_mask

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight * self.diagonal_mask, self.bias)


class Refine_LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, mean, std, num_classes=3, i_ratio=0.9, h_ratio=0.9):
        super(Refine_LSTM, self).__init__()
        self.input_size = input_size
        self.output_size=output_size
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.i_ratio=i_ratio # dropout ratio of input = 1 - i_ratio
        self.h_ratio=h_ratio # dropout ratio of hidden state = 1 - h_ratio
        self.pi = torch.Tensor([np.pi]).cuda()
        self.mean=mean
        self.std=std
        # Decay weights
        self.w_dg_x = Linear_diagonal_weight(input_size, input_size)
        self.w_dg_h = torch.nn.Linear(input_size, hidden_size, bias=True)

        # x weights
        self.w_x = torch.nn.Linear(hidden_size, input_size, bias=True)
        self.w_xz = Linear_diagonal_weight_z(input_size, input_size)

        #beta weight
        self.w_b_dg = torch.nn.Linear(input_size, input_size, bias=True)

        # i weights
        self.w_ui = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_hi = torch.nn.Linear(hidden_size, hidden_size, bias=True)        

        # c weights
        self.w_uc = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_hc = torch.nn.Linear(hidden_size, hidden_size, bias=True) 

        # o weights
        self.w_uo = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_ho = torch.nn.Linear(hidden_size, hidden_size, bias=True) 

        # h weights
        self.w_yh = torch.nn.Linear(hidden_size, num_classes, bias=True)
        self.w_gh = torch.nn.Linear(hidden_size, output_size-num_classes, bias=True)

        # r beta weights
        self.w_br = torch.nn.Linear(input_size, input_size, bias=True)

        # f weights
        self.w_uf = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_hf = torch.nn.Linear(hidden_size, hidden_size, bias=True)

        # r weights
        self.w_ur = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.w_hr = torch.nn.Linear(hidden_size, hidden_size, bias=True)   

        self.reset_parameters()
      
    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def imputation_module(self, x_t, m_t, gamma_x, h_t):
        raise NotImplementedError

    def encoder_module(self, u_t, m_t, h_t, c_t):
        raise NotImplementedError

    def prediction_module(self, h_t):
        y_t = torch.softmax(self.w_yh(h_t), dim=1)
        bio_t = self.w_gh(h_t)
        return y_t, bio_t

    def dropout_mask(self, batch_size):
        dev = next(self.parameters()).device
        i_mask_bio = torch.ones(batch_size, self.input_size - self.num_classes, device=dev)
        i_mask_dia = torch.ones(batch_size, self.num_classes, device=dev)
        h_mask = torch.ones(batch_size, self.hidden_size, device=dev)
        if self.training:
            i_mask_bio.bernoulli_(self.i_ratio)
            h_mask.bernoulli_(self.h_ratio)
        i_mask = torch.cat((i_mask_dia, i_mask_bio), axis=1)
        return i_mask, h_mask

    def forward(self, X, Mask, Delta):
        device = next(self.parameters()).device
        out_cat_val_seq = torch.empty((X.shape[0]-1, X.shape[1], X.shape[2]))
        xhat_cat_val_seq = torch.empty((X.shape[0], X.shape[1], X.shape[2]))
        zhat_cat_val_seq = torch.empty((X.shape[0], X.shape[1], X.shape[2]))
        uhat_cat_val_seq = torch.empty((X.shape[0], X.shape[1], X.shape[2]))
        h_t = torch.zeros((X.shape[1],self.hidden_size), dtype=torch.float, device= device)
        c_t = torch.zeros((X.shape[1],self.hidden_size), dtype=torch.float, device= device)
        for i in range(len(X)):
            x_t = torch.tensor(X[i], dtype=torch.float, device=device)
            m_t = torch.tensor(Mask[i], dtype=torch.float, device=device)
            delta_t = torch.tensor(Delta[i], dtype=torch.float, device=device)     
            
            gamma_x = torch.exp(-1* torch.nn.functional.relu( self.w_dg_x(delta_t))) #torch.Size([128, 14])
            gamma_h = torch.exp(-1* torch.nn.functional.relu( self.w_dg_h(delta_t)))
            # Imputation Module
            u_t, u_t_hat, z_t_hat, x_t_hat = self.imputation_module(x_t, m_t, gamma_x, h_t)
            # Encoder Module
            h_t, c_t = self.encoder_module(u_t, m_t, gamma_h, h_t, c_t)
            # Prediction Module
            y_t, bio_t = self.prediction_module(h_t)

            if i < len(X) -1:
                out_cat_val_seq[i] = torch.cat((y_t, bio_t), axis=1)
            xhat_cat_val_seq[i] = x_t_hat
            zhat_cat_val_seq[i] = z_t_hat
            uhat_cat_val_seq[i] = u_t_hat

        return out_cat_val_seq, xhat_cat_val_seq, zhat_cat_val_seq, uhat_cat_val_seq #torch.stack(out_cat_val_seq)

    def predict(self, X, Mask):
        with torch.no_grad():
            device = next(self.parameters()).device
            out_cat_val_seq = torch.empty((X.shape[0]-1, X.shape[1], X.shape[2]), dtype=torch.float)
            h_t = torch.zeros((X.shape[1],self.hidden_size), dtype=torch.float, device= device)
            c_t = torch.zeros((X.shape[1],self.hidden_size), dtype=torch.float, device= device)
            u_seq = torch.empty((X.shape[0]-1, X.shape[1], X.shape[2]-self.num_classes))
            forecast_seq = torch.empty((X.shape[0]-1, X.shape[1], X.shape[2]-self.num_classes))
            x_last_full = torch.zeros((X.shape[1],self.input_size), dtype=torch.float, device= device)
            delta_t = torch.zeros((X.shape[1], self.input_size), device=device)
            for i in range(len(X)-1):
                
                gamma_x = torch.exp(-1* torch.nn.functional.relu( self.w_dg_x(delta_t)))
                gamma_h = torch.exp(-1* torch.nn.functional.relu( self.w_dg_h(delta_t)))

                x_t_hat = self.w_x(h_t)
                if i == 0:
                    x_t = torch.tensor(X[i], device=device).float()
                    m_t = torch.tensor(Mask[i], device=device).float()
                else:
                    x_t = x_last_full.cuda()
                    m_t = torch.zeros((len(x_t), self.input_size), device=device)

                # Imputation Module
                u_t, u_t_hat, z_t_hat, x_t_hat = self.imputation_module(x_t, m_t, gamma_x, h_t)
                # Encoder Module
                h_t, c_t = self.encoder_module(u_t, m_t, gamma_h, h_t, c_t)
                # Prediction Module
                y_t, bio_t = self.prediction_module(h_t)

                out_cat_val_seq[i] = torch.cat((y_t, bio_t), axis=1)
                x_last_full = out_cat_val_seq[i]
                argmax = torch.argmax(y_t, axis=1)
                for m in range(len(argmax)):
                    for n in range(self.num_classes):
                        if argmax[m] == n:
                            x_last_full[m, n] = 1.
                        else:
                            x_last_full[m, n] = 0.
                u_seq[i] = u_t[:,self.num_classes:]
                forecast_seq[i] = bio_t
                delta_t += 1
            # f_t = f_t.cpu().numpy()
            # print(f_t.shape)
            # plt.hist(f_t, bins = 50, facecolor='blue')
            # plt.axis([0, 1, 0, 120])
            # plt.show()
                
        return out_cat_val_seq, u_seq, forecast_seq
class Refine_LSTM_v1(Refine_LSTM): # my proposal : apply refine gate into Encoder Module
    def imputation_module(self, x_t, m_t, gamma_x, h_t):
        x_t = x_t*self.dropout_mask(x_t.shape[0])[0] # dropout
        m_t = m_t*self.dropout_mask(m_t.shape[0])[0] # dropout

        x_t_hat = self.w_x(h_t) 
        x_t[x_t != x_t] = 0
        x_t_c = m_t*x_t + (1-m_t)*x_t_hat 
        z_t_hat = self.w_xz(x_t_c)
        beta = torch.sigmoid(self.w_b_dg(gamma_x*m_t))
        u_t_hat = beta*z_t_hat + (1-beta)*x_t_hat  
        u_t = m_t*x_t + (1-m_t)*u_t_hat    
        return u_t, u_t_hat, z_t_hat, x_t_hat

    def encoder_module(self, u_t, m_t, gamma_h, h_t, c_t):
        h_t = h_t*self.dropout_mask(h_t.shape[0])[1] # dropout

        h_t_hat = gamma_h*h_t
        f_t = torch.sigmoid(self.w_uf(u_t) + self.w_hf(h_t_hat))
        r_t = torch.sigmoid(self.w_ur(u_t) + self.w_hr(h_t_hat))
        g_t = f_t + torch.sin(f_t*self.pi)*torch.cos(r_t*self.pi)/self.pi
        c_t_hat = torch.tanh(self.w_uc(u_t) + self.w_hc(h_t_hat))
        c_t = g_t*c_t + (1-g_t)*c_t_hat
        o_t = torch.sigmoid(self.w_uo(u_t) + self.w_ho(h_t_hat))
        h_t = o_t*torch.tanh(c_t)      
        return h_t, c_t

class Refine_LSTM_v2(Refine_LSTM): # my proposal : Apply refine gate in Imputation and Encoder Module
    def imputation_module(self, x_t, m_t, gamma_x, h_t):
        x_t = x_t*self.dropout_mask(x_t.shape[0])[0] # dropout
        m_t = m_t*self.dropout_mask(m_t.shape[0])[0] # dropout

        x_t_hat = self.w_x(h_t)
        x_t[x_t != x_t] = 0
        x_t_c = m_t*x_t + (1-m_t)*x_t_hat
        z_t_hat = self.w_xz(x_t_c)
        beta = torch.sigmoid(self.w_b_dg(gamma_x*m_t))
        refine = torch.tanh(self.w_br(gamma_x*m_t))
        k = beta + torch.sin(beta*self.pi)*refine/self.pi
        u_t_hat = (1-k)*z_t_hat + k*x_t_hat
        u_t = m_t*x_t + (1-m_t)*u_t_hat   
        return u_t, u_t_hat, z_t_hat, x_t_hat

    def encoder_module(self, u_t, m_t, gamma_h, h_t, c_t):
        h_t = h_t*self.dropout_mask(h_t.shape[0])[1] # dropout

        h_t_hat = gamma_h*h_t
        f_t = torch.sigmoid(self.w_uf(u_t) + self.w_hf(h_t_hat))
        r_t = torch.sigmoid(self.w_ur(u_t) + self.w_hr(h_t_hat))
        g_t = f_t + torch.sin(f_t*self.pi)*torch.cos(r_t*self.pi)/self.pi
        c_t_hat = torch.tanh(self.w_uc(u_t) + self.w_hc(h_t_hat))
        c_t = g_t*c_t + (1-g_t)*c_t_hat
        o_t = torch.sigmoid(self.w_uo(u_t) + self.w_ho(h_t_hat))
        h_t = o_t*torch.tanh(c_t)      
        return h_t, c_t


