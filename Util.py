

import numpy as np



class MatVecMul(object):
    
    @staticmethod
    def fwd_pass(W, v):

        z = np.dot(W,v)
   
        return z
    
    @staticmethod
    def back_pass(dz, W, v):

        dW = np.outer(dz,v)
        
        dv = np.dot(W.T,dz)
        
        return dW, dv



class VecMatMul(object):
    
    @staticmethod
    def fwd_pass(v, W):
        
        z = np.dot(v,W)
        
        return z
    
    @staticmethod
    def back_pass(dz, v, W):
        
        dv = np.dot(W, dz)
        
        dW = np.outer(v, dz)
        
        return dv, dW



class OuterProduct(object):
    
    @staticmethod
    def fwd_pass(U, V):
        
        return np.outer(U,V)
    
    @staticmethod
    def back_pass(dO, U, V):
        
        dU = np.sum(dO*V,axis=1)
        
        dV = np.sum(dO*U.reshape(-1,1),axis=0)
        
        return dU, dV



class Negate(object):
    
    @staticmethod
    def fwd_pass(U):
        
        return 1-U
    
    @staticmethod
    def back_pass(dU):
        
        return -dU


class HadamardProduct(object):
    
    @staticmethod
    def fwd_pass(U,V):
        
        return U*V
    
    @staticmethod
    def back_pass(dO, U, V):
        
        return dO*V, dO*U


class Add(object):
  
    @staticmethod
    def fwd_pass(A,B):    
        
        return A+B
    
    @staticmethod
    def back_pass(dC):
        
        return dC, dC



class Sigmoid(object):
    
    def __init__(self):
        pass
    
    def sigmoid(z):
        
        return 1.0/(1.0+np.exp(-z))
  
    def fwd_pass(self, Q):

        self.Q = Q
        
        return Sigmoid.sigmoid(self.Q)
    
    def back_pass(self, dz):
        
        return dz*Sigmoid.sigmoid(self.Q)*(1-Sigmoid.sigmoid(self.Q))



class ReLU(object):
    
    def __init__(self):
        pass
    
    def ReLU_unit(z):
        
        zr = z.copy()
        
        zr[zr<0] = 0
        
        return zr
    
    def fwd_pass(self, z):
        
        self.z = z

        return ReLU.ReLU_unit(self.z)
    
    def back_pass(self, dz):

        if isinstance(dz, float):
            
            dz_relu = 0 if self.z<=0 else dz
        
        else:
            dz_relu = dz.copy()

            dz_relu[self.z<=0] = 0
        
        return dz_relu



def Clip(arr):
    '''
    Hard-coded lower and upper bounds as per paper
    '''
    return np.clip(arr,-10,10)


class Softmax(object):
    
    def __init__(self):
        pass
        
    def fwd_pass(self,z):
        
        self.z_exp = np.exp(z)
        
        self.norm = np.sum(self.z_exp)
        
        self.y = self.z_exp/self.norm
        
        return self.y
    
    def back_pass(self, dz):
        
        jac = -np.outer(self.y,self.y)
        
        jac.ravel()[:jac.shape[1]**2:jac.shape[1]+1] = self.y*(1-self.y)
        
        return np.dot(jac.T,dz)

class ScalarMul(object):
    
    @staticmethod
    def fwd_pass(a, z):
        
        return a*z
    
    @staticmethod
    def back_pass(d_az, a, z):
        
        da = np.sum(d_az*z)
        
        dz = a*d_az
        
        return da, dz


#Sequence generation
def Copy_seq_gen(vector_size=8, seq_length=1):

    copy_seq = np.random.randint(2,size=(vector_size,seq_length))

    input_sequence = np.zeros((vector_size+2,seq_length*2+2))
    output_sequence = np.zeros((vector_size+2,seq_length*2+2))

    input_sequence[-2,0]=1
    input_sequence[-1,seq_length+1]=1
    input_sequence[:-2,1:seq_length+1] = copy_seq

    output_sequence[:-2,seq_length+2:] = copy_seq
    
    return input_sequence, output_sequence


class CrossEntropyLossSigmoid(object):
    
    @staticmethod
    def fwd_pass(Y,t):

        loss = -np.sum(t*np.log(Y)+(1-t)*np.log(1-Y))
    
        return loss
    
    @staticmethod
    def back_pass(Y,t):

        return (Y-t)/(Y*(1-Y))

