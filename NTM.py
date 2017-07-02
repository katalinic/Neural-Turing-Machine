import numpy as np
import Controller
import AddressMech
import MemOps
import Util

class NTM(object):
    
    def __init__(self, H=100, N=128, M=20, shift_parameter_size=3, learning_rate=1e-4, decay_param=0.95, momentum=0.9):

        self.learning_rate = learning_rate
        self.decay_param = decay_param
        self.momentum=momentum
        
        '''
        Xavier weight initialisation
        '''
        self.read_weight_dict = {
            'w_r_k' : np.random.randn(M,H)/np.sqrt(M+H),
            'w_r_beta': np.random.randn(1,H)/np.sqrt(1+H),
            'w_r_g': np.random.randn(1,H)/np.sqrt(1+H),
            'w_r_s': np.random.randn(shift_parameter_size,H)/np.sqrt(shift_parameter_size+H),
            'w_r_gamma': np.random.randn(1,H)/np.sqrt(1+H),
            'b_r_k' : np.zeros(M),
            'b_r_beta': 0,
            'b_r_g': 0,
            'b_r_s': np.zeros(shift_parameter_size),
            'b_r_gamma': 0
        }

        self.write_weight_dict = {
            'w_w_k' : np.random.randn(M,H)/np.sqrt(M+H),
            'w_w_beta': np.random.randn(1,H)/np.sqrt(1+H),
            'w_w_g': np.random.randn(1,H)/np.sqrt(1+H),
            'w_w_s': np.random.randn(shift_parameter_size,H)/np.sqrt(shift_parameter_size+H),
            'w_w_gamma': np.random.randn(1,H)/np.sqrt(1+H),
            'w_w_e': np.random.randn(M,H)/np.sqrt(M+H),
            'w_w_a': np.random.randn(M,H)/np.sqrt(M+H),
            'b_w_k' : np.zeros(M),
            'b_w_beta': 0,
            'b_w_g': 0,
            'b_w_s': np.zeros(shift_parameter_size),
            'b_w_gamma': 0,
            'b_w_e': np.zeros(M),
            'b_w_a': np.zeros(M)
        }
        
        '''
        Weight and bias for read to controller transition
        '''
        self.w_r_h = np.random.randn(H,M)/np.sqrt(H+M)
        self.b_r_h = np.zeros(H)
        self.rms_grad_b_r_h = np.zeros_like(self.b_r_h)
        
        self.s_w_r_h = np.zeros((H,M))
        self.s_b_r_h = np.zeros(H)
        self.rms_grad_w_r_h = np.zeros_like(self.w_r_h)	
        
        self.rms_w_r_h = np.zeros_like(self.w_r_h)
        self.rms_b_r_h = np.zeros_like(self.b_r_h)

        self.moment_w_r_h = np.zeros_like(self.w_r_h)
        self.moment_b_r_h = np.zeros_like(self.b_r_h)
        
        '''
        Memory unit 
        '''
        self.Memory = np.zeros((1,N,M))
        self.d_Memory = np.zeros((1,N,M))

        '''
        Initialising previous values as necessary.
        '''
        self.prev_read = np.zeros((1,M))
        self.prev_write_weight = Util.f_softmax(np.random.rand(1,N))
        self.prev_read_weight = Util.f_softmax(np.random.rand(1,N))

        '''
        To store dicts of read and write head parameters through time
        k, beta, g, s, gamma
        k, beta, g, s, gamma, e, a
        '''
        self.Read_Parameters = []
        self.Write_Parameters = []
        
        '''
        Initialising deltas for previous (really t+1) values as necessary.
        '''
        self.delta_prev_read = np.zeros((1,M))
        self.delta_prev_write_weight = np.zeros((1,N))
        self.delta_prev_read_weight = np.zeros((1,N))
        
        self.delta_controller_write = np.zeros((1,H))
        self.delta_controller_read = np.zeros((1,H))
        
        '''
        Deltas for everything else
        '''
        self.s_read_weights = {
            'w_r_k' : 0,
            'w_r_beta': 0,
            'w_r_g': 0,
            'w_r_s': 0,
            'w_r_gamma': 0,
            'b_r_k' : 0,
            'b_r_beta': 0,
            'b_r_g': 0,
            'b_r_s': 0,
            'b_r_gamma': 0
        }

        self.s_write_weights = {
            'w_w_k' : 0,
            'w_w_beta': 0,
            'w_w_g': 0,
            'w_w_s': 0,
            'w_w_gamma': 0,
            'w_w_e': 0,
            'w_w_a': 0,
            'b_w_k' : 0,
            'b_w_beta': 0,
            'b_w_g': 0,
            'b_w_s': 0,
            'b_w_gamma': 0,
            'b_w_e': 0,
            'b_w_a': 0
        }

        self.rms_grad_read_weights = {
            'w_r_k' : 0,
            'w_r_beta': 0,
            'w_r_g': 0,
            'w_r_s': 0,
            'w_r_gamma': 0,
            'b_r_k' : 0,
            'b_r_beta': 0,
            'b_r_g': 0,
            'b_r_s': 0,
            'b_r_gamma': 0}

        self.rms_grad_write_weights = {
            'w_w_k' : 0,
            'w_w_beta': 0,
            'w_w_g': 0,
            'w_w_s': 0,
            'w_w_gamma': 0,
            'w_w_e': 0,
            'w_w_a': 0,
            'b_w_k' : 0,
            'b_w_beta': 0,
            'b_w_g': 0,
            'b_w_s': 0,
            'b_w_gamma': 0,
            'b_w_e': 0,
            'b_w_a': 0
        }

        self.rms_read_weights = {
            'w_r_k' : 0,
            'w_r_beta': 0,
            'w_r_g': 0,
            'w_r_s': 0,
            'w_r_gamma': 0,
            'b_r_k' : 0,
            'b_r_beta': 0,
            'b_r_g': 0,
            'b_r_s': 0,
            'b_r_gamma': 0
        }

        self.rms_write_weights = {
            'w_w_k' : 0,
            'w_w_beta': 0,
            'w_w_g': 0,
            'w_w_s': 0,
            'w_w_gamma': 0,
            'w_w_e': 0,
            'w_w_a': 0,
            'b_w_k' : 0,
            'b_w_beta': 0,
            'b_w_g': 0,
            'b_w_s': 0,
            'b_w_gamma': 0,
            'b_w_e': 0,
            'b_w_a': 0
        }

        self.moment_read_weights = {
            'w_r_k' : 0,
            'w_r_beta': 0,
            'w_r_g': 0,
            'w_r_s': 0,
            'w_r_gamma': 0,
            'b_r_k' : 0,
            'b_r_beta': 0,
            'b_r_g': 0,
            'b_r_s': 0,
            'b_r_gamma': 0
        }

        self.moment_write_weights = {
            'w_w_k' : 0,
            'w_w_beta': 0,
            'w_w_g': 0,
            'w_w_s': 0,
            'w_w_gamma': 0,
            'w_w_e': 0,
            'w_w_a': 0,
            'b_w_k' : 0,
            'b_w_beta': 0,
            'b_w_g': 0,
            'b_w_s': 0,
            'b_w_gamma': 0,
            'b_w_e': 0,
            'b_w_a': 0
        }

        '''
        Store controller values
        '''
        self.controller_time = np.zeros((1,H))

        self.ReadHead = []
        self.WriteHead = []
        self.ReadHeadParamGen = []
        self.WriteHeadParamGen = []

    def fwd_pass(self, controller):
   
        self.ReadHead.append(AddressMech.ReadHead())
        self.WriteHead.append(AddressMech.WriteHead())
        self.ReadHeadParamGen.append(Controller.read_head_parameters())
        self.WriteHeadParamGen.append(Controller.write_head_parameters())
        
        r_to_c_input = Util.MatVecMul.fwd_pass(self.w_r_h,self.prev_read[-1])
        r_to_c = Util.Add.fwd_pass(r_to_c_input,self.b_r_h)
        
        new_controller_state = controller + r_to_c
        
        #Generate parameters
        read_params = self.ReadHeadParamGen[-1].fwd_pass(new_controller_state, self.read_weight_dict)
        write_params = self.WriteHeadParamGen[-1].fwd_pass(new_controller_state, self.write_weight_dict)
        
        #Append parameters to respective lists
        self.Read_Parameters.append(read_params)
        self.Write_Parameters.append(write_params)
        
        #Using these parameters, perform write and read operations, and get reading and writing weights
        w_w, M_out = self.WriteHead[-1].fwd_pass(self.Memory[-1], self.prev_write_weight[-1], write_params)
        w_r, r = self.ReadHead[-1].fwd_pass(self.Memory[-1], self.prev_read_weight[-1], read_params)
        
        #Update Memory
        self.Memory = np.append(self.Memory,M_out[np.newaxis,:,:],axis=0)
        
        #Update previous 
        self.prev_read = np.vstack((self.prev_read,r))
        self.prev_write_weight = np.vstack((self.prev_write_weight,w_w))
        self.prev_read_weight = np.vstack((self.prev_read_weight,w_r))
        
        #Store controller values
        self.controller_time = np.vstack((self.controller_time,new_controller_state)) 
        
        return new_controller_state
    
    def back_pass(self, delta_controller_out): 

        d_controller = delta_controller_out
    
        #Backprop read head
        d_M_read, d_prev_read_weight, d_read_params = self.ReadHead[-1].back_pass(
            self.delta_prev_read[-1], self.delta_prev_read_weight[-1], self.Memory[-2],
            self.prev_read_weight[-2],self.Read_Parameters[-1])
                               
        self.delta_prev_read_weight = np.vstack((self.delta_prev_read_weight,d_prev_read_weight))
        
        delta_read_weight_dict, d_controller_read = self.ReadHeadParamGen[-1].back_pass(d_read_params, 
                                                                self.controller_time[-1],self.read_weight_dict)
                                         
        for key in self.s_read_weights.keys():
            self.s_read_weights[key] += delta_read_weight_dict[key]
            
        #Backprop write head
        d_M_write, d_prev_write_weight, d_write_params = self.WriteHead[-1].back_pass(
            self.d_Memory[-1], self.delta_prev_write_weight[-1]	, self.Memory[-2], 
            self.prev_write_weight[-2],self.Write_Parameters[-1])
        
        self.delta_prev_write_weight = np.vstack((self.delta_prev_write_weight,d_prev_write_weight))
        
        
        self.d_Memory = np.append(self.d_Memory,d_M_write[np.newaxis,:,:]+d_M_read[np.newaxis,:,:],axis=0)
        self.d_Memory[-1] += self.d_Memory[-2]

        delta_write_weight_dict, d_controller_write = self.WriteHeadParamGen[-1].back_pass(
            d_write_params, self.controller_time[-1],self.write_weight_dict)
        
        for key in self.s_write_weights.keys():
            self.s_write_weights[key] += delta_write_weight_dict[key]
        
        d_controller += d_controller_read + d_controller_write

        d_read_to_controller_input, d_read_to_controller_bias = Util.Add.back_pass(d_controller)
        delta_w_r_h, d_read = Util.MatVecMul.back_pass(d_read_to_controller_input, self.w_r_h, self.prev_read[-2])
        self.delta_prev_read = np.vstack((self.delta_prev_read,d_read))
        	
        self.s_w_r_h += delta_w_r_h
        self.s_b_r_h += d_read_to_controller_bias
        
        '''
        Deletion as necessary 
        '''
        self.Memory = np.delete(self.Memory, -1, 0)
        
        self.controller_time = np.delete(self.controller_time, -1, 0)
        
        self.Read_Parameters = self.Read_Parameters[:-1]
        self.Write_Parameters = self.Write_Parameters[:-1]
        
        self.prev_read_weight = np.delete(self.prev_read_weight, -1, 0)
        self.prev_write_weight = np.delete(self.prev_write_weight, -1, 0)
        
        self.prev_read = np.delete(self.prev_read, -1, 0)

        self.ReadHead = self.ReadHead[:-1]
        self.WriteHead = self.WriteHead[:-1]
        self.ReadHeadParamGen = self.ReadHeadParamGen[:-1]
        self.WriteHeadParamGen = self.WriteHeadParamGen[:-1]
        
        return d_controller
    
    def weight_update(self):
        
   
        self.s_w_r_h = Util.Clip(self.s_w_r_h)
        self.s_b_r_h = Util.Clip(self.s_b_r_h)
        
        self.rms_w_r_h = self.decay_param*self.rms_w_r_h+(1-self.decay_param)*self.s_w_r_h**2
        self.rms_b_r_h = self.decay_param*self.rms_b_r_h+(1-self.decay_param)*self.s_b_r_h**2

        self.moment_w_r_h = self.decay_param*self.moment_w_r_h+(1-self.decay_param)*self.s_w_r_h
        self.moment_b_r_h = self.decay_param*self.moment_b_r_h+(1-self.decay_param)*self.s_b_r_h
        
        self.rms_grad_w_r_h = self.momentum*self.rms_grad_w_r_h - self.learning_rate*self.s_w_r_h/np.sqrt(self.rms_w_r_h-self.moment_w_r_h**2+1e-4)
        self.rms_grad_b_r_h = self.momentum*self.rms_grad_b_r_h - self.learning_rate*self.s_b_r_h/np.sqrt(self.rms_b_r_h-self.moment_b_r_h**2+1e-4)	
        
        self.w_r_h += self.rms_grad_w_r_h
        self.b_r_h += self.rms_grad_b_r_h
        
        
        for key in self.s_read_weights.keys():
            self.rms_read_weights[key] = self.decay_param*self.rms_read_weights[key] +             (1-self.decay_param)*Util.Clip(self.s_read_weights[key])**2
        
        for key in self.s_write_weights.keys():
            self.rms_write_weights[key] = self.decay_param*self.rms_write_weights[key] +             (1-self.decay_param)*Util.Clip(self.s_write_weights[key])**2

        for key in self.s_read_weights.keys():
            self.moment_read_weights[key] = self.decay_param*self.moment_read_weights[key] +             (1-self.decay_param)*Util.Clip(self.s_read_weights[key])
        
        for key in self.s_write_weights.keys():
            self.moment_write_weights[key] = self.decay_param*self.moment_write_weights[key] +             (1-self.decay_param)*Util.Clip(self.s_write_weights[key])
            
        for key in self.read_weight_dict.keys():
            self.rms_grad_read_weights[key] = self.momentum*self.rms_grad_read_weights[key] - \
		self.learning_rate*Util.Clip(self.s_read_weights[key])/np.sqrt(self.rms_read_weights[key]-self.moment_read_weights[key]**2+1e-4)
        
        for key in self.write_weight_dict.keys():
            self.rms_grad_write_weights[key] = self.momentum*self.rms_grad_write_weights[key] - \
		self.learning_rate*Util.Clip(self.s_write_weights[key])/np.sqrt(self.rms_write_weights[key]-self.moment_write_weights[key]**2+1e-4)

        for key in self.read_weight_dict.keys():
            self.read_weight_dict[key] += self.rms_grad_read_weights[key]

        for key in self.write_weight_dict.keys():
            self.write_weight_dict[key] += self.rms_grad_write_weights[key]

        return None
    
    def reset(self, N=128, M=20, H=100, shift_parameter_size=3):
        
        self.Memory = np.zeros((1,N,M))
        self.d_Memory = np.zeros((1,N,M))

        self.s_w_r_h = np.zeros((H,M))
        self.s_b_r_h = np.zeros(H)
        
        self.prev_read = np.zeros((1,M))
        self.prev_write_weight = Util.f_softmax(np.random.rand(1,N))
        self.prev_read_weight = Util.f_softmax(np.random.rand(1,N))

        self.Read_Parameters = []
        self.Write_Parameters = []
     
        self.delta_prev_read = np.zeros((1,M))
        self.delta_prev_write_weight = np.zeros((1,N))
        self.delta_prev_read_weight = np.zeros((1,N))
        
        self.delta_controller_write = np.zeros((1,H))
        self.delta_controller_read = np.zeros((1,H))

        self.ReadHead = []
        self.WriteHead = []
        self.ReadHeadParamGen = []
        self.WriteHeadParamGen = []
        
        self.s_read_weights = {
            'w_r_k' : 0,
            'w_r_beta': 0,
            'w_r_g': 0,
            'w_r_s': 0,
            'w_r_gamma': 0,
            'b_r_k' : 0,
            'b_r_beta': 0,
            'b_r_g': 0,
            'b_r_s': 0,
            'b_r_gamma': 0
        }

        self.s_write_weights = {
            'w_w_k' : 0,
            'w_w_beta': 0,
            'w_w_g': 0,
            'w_w_s': 0,
            'w_w_gamma': 0,
            'w_w_e': 0,
            'w_w_a': 0,
            'b_w_k' : 0,
            'b_w_beta': 0,
            'b_w_g': 0,
            'b_w_s': 0,
            'b_w_gamma': 0,
            'b_w_e': 0,
            'b_w_a': 0
        }

        self.controller_time = np.zeros((1,H))

        return None
