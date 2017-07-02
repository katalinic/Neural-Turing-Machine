import numpy as np
import Util
import MemOps

class cosineSim(object):
    
    def __init__(self):
        pass
    
    def fwd_pass(self, k, M):
        
        self.dot = np.dot(M,k)
        
        self.k_norm = np.sqrt(np.sum(k**2))
        
        self.M_norm = np.sqrt(np.sum(M**2,axis=1))
        
        self.cos_sim = self.dot/(self.k_norm*self.M_norm+1e-3)
        
        return self.cos_sim
    
    def back_pass(self, dw, k, M):
        
        dk = M/(self.k_norm*self.M_norm.reshape(-1,1)+1e-3)-np.outer(self.cos_sim,k)/self.k_norm**2

        dkw = np.dot(dk.T,dw)
        
        dm = k/(self.k_norm*self.M_norm.reshape(-1,1)+1e-3)-self.cos_sim.reshape(-1,1)*M/((self.M_norm**2).reshape(-1,1)+1e-3)
        
        dmw = dw.reshape(-1,1)*dm
        
        return dkw, dmw

class ContentAddress(object):
    
    def __init__(self):
        self.cossim = cosineSim()
        self.sfmax = Util.Softmax()
    
    def fwd_pass(self, k, M, beta):
        
        self.w_cosine = self.cossim.fwd_pass(k, M)
        
        w_beta = Util.ScalarMul.fwd_pass(beta, self.w_cosine)
        
        w_softmax = self.sfmax.fwd_pass(w_beta)

        return w_softmax
    
    def back_pass(self, d_w_ca, k, M, beta):
        
        d_w_softmax = self.sfmax.back_pass(d_w_ca)
        
        d_beta, d_w_cosine = Util.ScalarMul.back_pass(d_w_softmax, beta, self.w_cosine)
        
        d_k, d_M = self.cossim.back_pass(d_w_cosine, k, M)
        
        return d_k, d_M, d_beta

class Interpolation(object):
    
    def __init__(self):
        pass
    
    def fwd_pass(self, w_c, w_prev, g):
        
        w_g = g*w_c+(1-g)*w_prev
        
        return w_g
    
    def back_pass(self, dw_g, w_prev, w_c, g):
        
        dw_prev = (1-g)*dw_g
        
        dw_c = g*dw_g
        
        dg = np.sum(dw_g*(w_c-w_prev))
        
        return dw_c, dw_prev, dg

class Convolution_shift(object):
    
    def __init__(self):
        pass
    
    def fwd_pass(self, wg, s):
        
        left = np.hstack((wg[1:],wg[0]))
        right = np.hstack((wg[-1],wg[:-1]))
        self.convW = np.vstack((left,wg,right)).T
        
        return np.dot(self.convW,s)
    
    def back_pass(self, dw_s, wg, s):
        
        ds = np.dot(self.convW.T,dw_s)

        left_back = np.hstack((dw_s[-1],dw_s[:-1]))
        right_back = np.hstack((dw_s[1:],dw_s[0]))
        final_back = np.vstack((left_back,dw_s,right_back)).T

        dw_g = np.dot(final_back,s)
        
        return dw_g, ds

class Sharpening(object):
    
    def __init__(self):
        pass
    
    def fwd_pass(self,ws, gamma):
        
        self.pw = ws**gamma
        
        self.norm = np.sum(self.pw)
        
        self.wgamma = self.pw/self.norm
        
        return self.wgamma
    
    def back_pass(self,d_wgamma, ws, gamma):
        
        jac_ws = -gamma/ws*np.outer(self.wgamma,self.wgamma)
        
        jac_ws.ravel()[:jac_ws.shape[1]**2:jac_ws.shape[1]+1] = gamma/ws*self.wgamma*(1-self.wgamma)
        
        d_ws = np.dot(jac_ws.T,d_wgamma)
        
        d_g = self.wgamma*(np.log(ws)-np.sum(self.pw*np.log(ws))/self.norm)
        
        d_gamma = np.sum(d_g*d_wgamma)
        
        return d_gamma, d_ws

class AddressMechanism(object):
    
    def __init__(self):
        self.CA = ContentAddress()
        self.Intr = Interpolation()
        self.Conv = Convolution_shift()
        self.Sharp = Sharpening()
    
    def fwd_pass(self, M, w_prev, k, beta, g, s, gamma):
        
        self.w_ca = self.CA.fwd_pass(k, M, beta)
        
        self.w_g = self.Intr.fwd_pass(self.w_ca, w_prev, g)
        
        self.w_cs = self.Conv.fwd_pass(self.w_g, s)
        
        w_gamma = self.Sharp.fwd_pass(self.w_cs, gamma)
        
        return w_gamma
    
    def back_pass(self, dw, M, w_prev, k, beta, g, s, gamma):
        
        d_gamma, d_w_cs = self.Sharp.back_pass(dw, self.w_cs, gamma)
        
        d_w_g, d_s = self.Conv.back_pass(d_w_cs, self.w_g, s)
        
        d_w_ca, d_w_prev, d_g = self.Intr.back_pass(d_w_g, w_prev, self.w_ca, g)
        
        d_k, d_M, d_beta = self.CA.back_pass(d_w_ca, k, M, beta)
        
        return d_M, d_w_prev, d_k, d_beta, d_g, d_s, d_gamma

class ReadHead(object):
    
    def __init__(self):
        self.AM = AddressMechanism()
        self.RM = MemOps.ReadMem()
    
    def fwd_pass(self, M, w_prev, read_params):
        
        k, beta, g, s, gamma = read_params.values()
        
        self.w_r = self.AM.fwd_pass(M, w_prev, k, beta, g, s, gamma)
        
        r = self.RM.fwd_pass(self.w_r, M)
        
        return self.w_r, r
    
    def back_pass(self, d_r, d_w_r_prev, M, w_prev, read_params):
        
        k, beta, g, s, gamma = read_params.values()
        
        d_w_r, d_M_r = self.RM.back_pass(d_r, self.w_r, M)
        
        d_M_a, d_w_prev, d_k, d_beta, d_g, d_s, d_gamma = self.AM.back_pass(d_w_r+d_w_r_prev, M, w_prev, k, beta, g, s, gamma)
        
        d_read_params = {
            'd_k': d_k,
            'd_beta': d_beta,
            'd_g': d_g,
            'd_s': d_s,
            'd_gamma': d_gamma
        }
        
        return d_M_a+d_M_r, d_w_prev, d_read_params

class WriteHead(object):
    
    def __init__(self):
        self.AM = AddressMechanism()
        self.WM = MemOps.WriteMem()
    
    def fwd_pass(self, M, w_prev, write_params):
        
        k, beta, g, s, gamma, e, a = write_params.values()
        
        self.w_w = self.AM.fwd_pass(M, w_prev, k, beta, g, s, gamma)
        
        M_out = self.WM.fwd_pass(M, self.w_w, a, e)
        
        return self.w_w, M_out
    
    def back_pass(self, d_M, d_w_w_prev, M, w_prev, write_params):
        
        k, beta, g, s, gamma, e, a = write_params.values()
        
        d_M_writing, d_w_w, d_a, d_e = self.WM.back_pass(d_M, M, self.w_w, a, e)
        
        d_M_a, d_w_prev, d_k, d_beta, d_g, d_s, d_gamma = self.AM.back_pass(d_w_w+d_w_w_prev, M, w_prev, k, beta, g, s, gamma)
        
        d_write_params = {
            'd_k': d_k,
            'd_beta': d_beta,
            'd_g': d_g,
            'd_s': d_s,
            'd_gamma': d_gamma,
            'd_e': d_e,
            'd_a': d_a
        }
        
        return d_M_writing+d_M_a, d_w_prev, d_write_params
