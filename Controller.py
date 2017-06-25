
import numpy as np
import Util



class read_head_parameters(object):
    
    def __init__(self):
        self.k_act = Util.ReLU()
        self.beta_act = Util.ReLU()
        self.g_act = Util.Sigmoid()
        self.s_act = Util.Softmax()
        self.gamma_act = Util.ReLU()
    
    def fwd_pass(self, cntlr, read_weights):
        
        #k
        k_input = Util.MatVecMul.fwd_pass(read_weights['w_r_k'],cntlr)
        k_plus_bias = Util.Add.fwd_pass(read_weights['b_r_k'],k_input)
        k = self.k_act.fwd_pass(k_plus_bias)
        
        #beta
        beta_input = Util.MatVecMul.fwd_pass(read_weights['w_r_beta'],cntlr)
        beta_plus_bias = Util.Add.fwd_pass(read_weights['b_r_beta'],beta_input)
        beta = self.beta_act.fwd_pass(beta_plus_bias)
        
        #g
        g_input = Util.MatVecMul.fwd_pass(read_weights['w_r_g'],cntlr)
        g_plus_bias = Util.Add.fwd_pass(read_weights['b_r_g'],g_input)
        g = self.g_act.fwd_pass(g_plus_bias)
        
        #s
        s_input = Util.MatVecMul.fwd_pass(read_weights['w_r_s'],cntlr)
        s_plus_bias = Util.Add.fwd_pass(read_weights['b_r_s'],s_input)
        s = self.s_act.fwd_pass(s_plus_bias)
        
        #gamma
        gamma_input = Util.MatVecMul.fwd_pass(read_weights['w_r_gamma'],cntlr)
        gamma_plus_bias = Util.Add.fwd_pass(read_weights['b_r_gamma'],gamma_input)
        gamma = 1+self.gamma_act.fwd_pass(gamma_plus_bias)
        
        read_param_dict = {
            'k': k,
            'beta': beta,
            'g': g,
            's': s,
            'gamma': gamma
        }
        
        return read_param_dict
    
    def back_pass(self, delta_read_weights, cntlr, read_weights):
        
        d_k, d_beta, d_g, d_s, d_gamma = delta_read_weights.values()
        
        #k
        d_k_out = self.k_act.back_pass(d_k)
        d_k_input, d_k_bias = Util.Add.back_pass(d_k_out)
        d_w_r_k, d_cntlr_k = Util.MatVecMul.back_pass(d_k_input, read_weights['w_r_k'],cntlr)
    
        #beta
        d_beta_out = self.beta_act.back_pass(d_beta)
        d_beta_input, d_beta_bias = Util.Add.back_pass(d_beta_out)
        d_w_r_beta, d_cntlr_beta = Util.MatVecMul.back_pass(d_beta_input, read_weights['w_r_beta'],cntlr)
        
        #g
        d_g_out = self.g_act.back_pass(d_g)
        d_g_input, d_g_bias = Util.Add.back_pass(d_g_out)
        d_w_r_g, d_cntlr_g = Util.MatVecMul.back_pass(d_g_input, read_weights['w_r_g'],cntlr)
    
        #s
        d_s_out = self.s_act.back_pass(d_s)
        d_s_input, d_s_bias = Util.Add.back_pass(d_s_out)
        d_w_r_s, d_cntlr_s = Util.MatVecMul.back_pass(d_s_input, read_weights['w_r_s'],cntlr)
        
        #gamma
        d_gamma_out = self.gamma_act.back_pass(d_gamma)
        d_gamma_input, d_gamma_bias = Util.Add.back_pass(d_gamma_out)
        d_w_r_gamma, d_cntlr_gamma = Util.MatVecMul.back_pass(d_gamma_input, read_weights['w_r_gamma'],cntlr)

        deltas_dict = {
                'w_r_k' : d_w_r_k,
                'w_r_beta': d_w_r_beta,
                'w_r_g': d_w_r_g,
                'w_r_s': d_w_r_s,
                'w_r_gamma': d_w_r_gamma,
                'b_r_k' : d_k_bias,
                'b_r_beta': d_beta_bias,
                'b_r_g': d_g_bias,
                'b_r_s': d_s_bias,
                'b_r_gamma': d_gamma_bias 
            }
        
        d_controller = d_cntlr_k+d_cntlr_beta+d_cntlr_g+d_cntlr_s+d_cntlr_gamma
        
        return deltas_dict, d_controller


class write_head_parameters(object):
    
    def __init__(self):
        self.k_act = Util.ReLU()
        self.beta_act = Util.ReLU()
        self.g_act = Util.Sigmoid()
        self.s_act = Util.Softmax()
        self.gamma_act = Util.ReLU()
        self.e_act = Util.Sigmoid()
        self.a_act = Util.ReLU()
    
    def fwd_pass(self, cntlr, write_weights):
        
        #k
        k_input = Util.MatVecMul.fwd_pass(write_weights['w_w_k'],cntlr)
        k_plus_bias = Util.Add.fwd_pass(write_weights['b_w_k'],k_input)
        k = self.k_act.fwd_pass(k_plus_bias)
        
        #beta
        beta_input = Util.MatVecMul.fwd_pass(write_weights['w_w_beta'],cntlr)
        beta_plus_bias = Util.Add.fwd_pass(write_weights['b_w_beta'],beta_input)
        beta = self.beta_act.fwd_pass(beta_plus_bias)
        
        #g
        g_input = Util.MatVecMul.fwd_pass(write_weights['w_w_g'],cntlr)
        g_plus_bias = Util.Add.fwd_pass(write_weights['b_w_g'],g_input)
        g = self.g_act.fwd_pass(g_plus_bias)
        
        #s
        s_input = Util.MatVecMul.fwd_pass(write_weights['w_w_s'],cntlr)
        s_plus_bias = Util.Add.fwd_pass(write_weights['b_w_s'],s_input)
        s = self.s_act.fwd_pass(s_plus_bias)
        
        #gamma
        gamma_input = Util.MatVecMul.fwd_pass(write_weights['w_w_gamma'],cntlr)
        gamma_plus_bias = Util.Add.fwd_pass(write_weights['b_w_gamma'],gamma_input)
        gamma = 1+self.gamma_act.fwd_pass(gamma_plus_bias)
        
        #e
        e_input = Util.MatVecMul.fwd_pass(write_weights['w_w_e'],cntlr)
        e_plus_bias = Util.Add.fwd_pass(write_weights['b_w_e'],e_input)
        e = self.e_act.fwd_pass(e_plus_bias)
        
        #a
        a_input = Util.MatVecMul.fwd_pass(write_weights['w_w_a'],cntlr)
        a_plus_bias = Util.Add.fwd_pass(write_weights['b_w_a'],a_input)
        a = self.a_act.fwd_pass(a_plus_bias)
        
        write_param_dict = {
            'k': k,
            'beta': beta,
            'g': g,
            's': s,
            'gamma': gamma,
            'e': e,
            'a': a
        }
        
        return write_param_dict
    
    def back_pass(self, delta_write_weights, cntlr, write_weights):
        
        d_k, d_beta, d_g, d_s, d_gamma, d_e, d_a = delta_write_weights.values()
        
        #k
        d_k_out = self.k_act.back_pass(d_k)
        d_k_input, d_k_bias = Util.Add.back_pass(d_k_out)
        d_w_w_k, d_cntlr_k = Util.MatVecMul.back_pass(d_k_input, write_weights['w_w_k'],cntlr)
    
        #beta
        d_beta_out = self.beta_act.back_pass(d_beta)
        d_beta_input, d_beta_bias = Util.Add.back_pass(d_beta_out)
        d_w_w_beta, d_cntlr_beta = Util.MatVecMul.back_pass(d_beta_input, write_weights['w_w_beta'],cntlr)
        
        #g
        d_g_out = self.g_act.back_pass(d_g)
        d_g_input, d_g_bias = Util.Add.back_pass(d_g_out)
        d_w_w_g, d_cntlr_g = Util.MatVecMul.back_pass(d_g_input, write_weights['w_w_g'],cntlr)
    
        #s
        d_s_out = self.s_act.back_pass(d_s)
        d_s_input, d_s_bias = Util.Add.back_pass(d_s_out)
        d_w_w_s, d_cntlr_s = Util.MatVecMul.back_pass(d_s_input, write_weights['w_w_s'],cntlr)
        
        #gamma
        d_gamma_out = self.gamma_act.back_pass(d_gamma)
        d_gamma_input, d_gamma_bias = Util.Add.back_pass(d_gamma_out)
        d_w_w_gamma, d_cntlr_gamma = Util.MatVecMul.back_pass(d_gamma_input, write_weights['w_w_gamma'],cntlr)
        
        #e
        d_e_out = self.e_act.back_pass(d_e)
        d_e_input, d_e_bias = Util.Add.back_pass(d_e_out)
        d_w_w_e, d_cntlr_e = Util.MatVecMul.back_pass(d_e_input, write_weights['w_w_e'],cntlr)
        
        #a
        d_a_out = self.a_act.back_pass(d_a)
        d_a_input, d_a_bias = Util.Add.back_pass(d_a_out)
        d_w_w_a, d_cntlr_a = Util.MatVecMul.back_pass(d_a_input, write_weights['w_w_a'],cntlr)

        deltas_dict = {
                'w_w_k' : d_w_w_k,
                'w_w_beta': d_w_w_beta,
                'w_w_g': d_w_w_g,
                'w_w_s': d_w_w_s,
                'w_w_gamma': d_w_w_gamma,
                'w_w_e': d_w_w_e,
                'w_w_a': d_w_w_a,
                'b_w_k' : d_k_bias,
                'b_w_beta': d_beta_bias,
                'b_w_g': d_g_bias,
                'b_w_s': d_s_bias,
                'b_w_gamma': d_gamma_bias,
                'b_w_e': d_e_bias,
                'b_w_a': d_a_bias
            }
        
        d_controller = d_cntlr_k+d_cntlr_beta+d_cntlr_g+d_cntlr_s+d_cntlr_gamma+d_cntlr_e + d_cntlr_a
        
        return deltas_dict, d_controller

