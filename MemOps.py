
import numpy as np
import Util


class ReadMem(object):
    
    def __init__(self):
        pass
    
    def fwd_pass(self, w, M):
        
        r = Util.VecMatMul.fwd_pass(w,M)
        
        return r
    
    def back_pass(self, dr, w, M):
        
        dw, dM = Util.VecMatMul.back_pass(dr, w, M)
        
        return dw, dM


class EraseMem(object):
    
    def __init__(self):
        pass
    
    def fwd_pass(self, M, w, e):
        
        op = Util.OuterProduct.fwd_pass(w,e)
        
        neg = Util.Negate.fwd_pass(op)
        
        self.hp = Util.HadamardProduct.fwd_pass(neg, M)
 
        return self.hp
    
    def back_pass(self, dM, M, w, e):
        
        d_hp, d_M_in = Util.HadamardProduct.back_pass(dM, self.hp, M)
        
        d_neg = Util.Negate.back_pass(d_hp)
        
        d_w, d_e = Util.OuterProduct.back_pass(d_neg, w, e)
        
        return d_e, d_w, d_M_in



class AddMem(object):
    
    def __init__(self):
        pass
    
    def fwd_pass(self, M,w,a):
                
        op = Util.OuterProduct.fwd_pass(w,a)

        ad = Util.Add.fwd_pass(op, M)
        
        return ad
        
    def back_pass(self, dM, w, a):
        
        d_op, d_M_in = Util.Add.back_pass(dM)
        
        d_w, d_a = Util.OuterProduct.back_pass(d_op, w, a)
        
        return d_a, d_w, d_M_in



class WriteMem(object):
    
    def __init__(self):
        pass
    
    def fwd_pass(self, M, w, a ,e):
        
        M_erase = EraseMem.fwd_pass(self, M, w, e)
        
        M_add = AddMem.fwd_pass(self, M_erase, w, a)
        
        return M_add
    
    def back_pass(self, dM, M_init, w, a, e):
        
        d_a, d_w_a, d_M_had = AddMem.back_pass(self, dM, w, a)
        
        d_e, d_w_e, d_M_init = EraseMem.back_pass(self, d_M_had, M_init, w, e)
        
        return d_M_init, d_w_e+d_w_a, d_a, d_e

