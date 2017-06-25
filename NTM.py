
import numpy as np
import Controller
import AddressMech
import MemOps
import Util



class NTM(object):
    
    def __init__(self, H=100, N=128, M=20, shift_parameter_size=3, learning_rate=-1e-4):
    
        self.learning_rate = learning_rate
        
        '''
        Xavier weight initialisation
        '''
        self.read_weight_dict = {
            'w_r_k' : np.random.randn(M,H)/np.sqrt(M+H),
            'w_r_beta': np.random.randn(1,H)/np.sqrt(1+H),
            'w_r_g': np.random.randn(1,H)/np.sqrt(1+H),
            'w_r_s': np.random.randn(shift_parameter_size,H)/np.sqrt(shift_parameter_size+H),
            'w_r_gamma': np.random.randn(1,H)/np.sqrt(1+H),
            'b_r_k' : np.random.randn(M)/np.sqrt(M),
            'b_r_beta': np.random.randn(1)/np.sqrt(1),
            'b_r_g': np.random.randn(1)/np.sqrt(1),
            'b_r_s': np.random.randn(shift_parameter_size)/np.sqrt(shift_parameter_size),
            'b_r_gamma': np.random.randn(1)/np.sqrt(1)
        }

        self.write_weight_dict = {
            'w_w_k' : np.random.randn(M,H)/np.sqrt(M+H),
            'w_w_beta': np.random.randn(1,H)/np.sqrt(1+H),
            'w_w_g': np.random.randn(1,H)/np.sqrt(1+H),
            'w_w_s': np.random.randn(shift_parameter_size,H)/np.sqrt(shift_parameter_size+H),
            'w_w_gamma': np.random.randn(1,H)/np.sqrt(1+H),
            'w_w_e': np.random.randn(M,H)/np.sqrt(M+H),
            'w_w_a': np.random.randn(M,H)/np.sqrt(M+H),
            'b_w_k' : np.random.randn(M)/np.sqrt(M),
            'b_w_beta': np.random.randn(1)/np.sqrt(1),
            'b_w_g': np.random.randn(1)/np.sqrt(1),
            'b_w_s': np.random.randn(shift_parameter_size)/np.sqrt(shift_parameter_size),
            'b_w_gamma': np.random.randn(1)/np.sqrt(1),
            'b_w_e': np.random.randn(M)/np.sqrt(M),
            'b_w_a': np.random.randn(M)/np.sqrt(M)
        }
        
        '''
        Weight and bias for read to controller transition
        '''
        self.w_r_h = np.random.randn(H,M)/np.sqrt(H+M)
        self.b_r_h = np.random.randn(H)/np.sqrt(H)
        
        self.d_w_r_h = np.zeros((H,M))
        self.d_b_r_h = np.zeros(H)
        
        '''
        Memory unit 
        Append as np.append(M1,M2[np.newaxis,:,:],axis=0)
        '''
        self.Memory = np.zeros((1,N,M))+0.1
        self.d_Memory = np.zeros((1,N,M))
        
        '''
        Initialising previous values as necessary. Vstack to keep track through time
        '''
        self.prev_read = np.zeros((1,M))
        self.prev_write_weight = np.zeros((1,N))
        self.prev_read_weight = np.zeros((1,N))
        
        '''
        To store dicts of read and write head parameters through time
        k, beta, g, s, gamma
        k, beta, g, s, gamma, e, a
        '''
        self.Read_Parameters = []
        self.Write_Parameters = []
        
        '''
        Initialising deltas for previous (really t+1) values as necessary. Vstack to keep track through time
        '''
        self.delta_prev_read = np.zeros((1,M))
        self.delta_prev_write_weight = np.zeros((1,N))
        self.delta_prev_read_weight = np.zeros((1,N))
        
        self.delta_controller_write = np.zeros((1,H))
        self.delta_controller_read = np.zeros((1,H))
        
        '''
        Deltas for everything else
        '''
        self.delta_read_weights = {
            'w_r_k' : 0,
            'w_r_beta': 0,
            'w_r_g': 0,
            'w_r_s': 0,
            'w_r_gamma': 0,
            'b_r_k' : 0,
            'b_r_beta': 0,
            'b_r_g': 0,
            'b_r_s': 0,
            'b_r_gamma': 0,
        }

        self.delta_write_weights = {
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
        
    def fwd_pass(self, controller):
        
        self.ReadHead = AddressMech.ReadHead()
        self.WriteHead = AddressMech.WriteHead()
        
        self.ReadHeadParamGen = Controller.read_head_parameters()
        self.WriteHeadParamGen = Controller.write_head_parameters()
        
        r_to_c_input = Util.MatVecMul.fwd_pass(self.w_r_h,self.prev_read[-1])
        r_to_c = Util.Add.fwd_pass(r_to_c_input,self.b_r_h)
        
        new_controller_state = controller + r_to_c
        
        #Generate parameters
        read_params = self.ReadHeadParamGen.fwd_pass(new_controller_state, self.read_weight_dict)
        write_params = self.WriteHeadParamGen.fwd_pass(new_controller_state, self.write_weight_dict)
        
        #Append parameters to respective lists
        self.Read_Parameters.append(read_params)
        self.Write_Parameters.append(write_params)
        
        #Using these parameters, perform write and read operations, and get reading and writing weights
        w_w, M_out = self.WriteHead.fwd_pass(self.Memory[-1], self.prev_write_weight[-1], write_params)
        w_r, r = self.ReadHead.fwd_pass(self.Memory[-1], self.prev_read_weight[-1], read_params)
        
        #Update Memory
        self.Memory = np.append(self.Memory,M_out[np.newaxis,:,:],axis=0)
        
        #Update previous 
        self.prev_read = np.vstack((self.prev_read,r))
        self.prev_write_weight = np.vstack((self.prev_write_weight,w_w))
        self.prev_read_weight = np.vstack((self.prev_read_weight,w_r))
        
        #Store controller values
        self.controller_time = np.vstack((self.controller_time,controller))
        
        return new_controller_state
    
    def back_pass(self, delta_controller_out):
        
        #Output to controller and controller t+1
        d_controller = delta_controller_out + self.delta_controller_write[-1]+self.delta_controller_read[-1]

        
        #Controller to read
        d_read_to_controller_input, d_read_to_controller_bias = Util.Add.back_pass(d_controller)
        delta_w_r_h, d_read = Util.MatVecMul.back_pass(d_read_to_controller_input, self.w_r_h, self.prev_read[-1])
        
        self.d_w_r_h += delta_w_r_h
        self.d_b_r_h += d_read_to_controller_bias
        
    
        #Backprop read head
        d_M_read, d_prev_read_weight, d_read_params = self.ReadHead.back_pass(
            d_read+self.delta_prev_read[-1], self.delta_prev_read_weight[-1], self.Memory[-1],
            self.prev_read_weight[-1],self.Read_Parameters[-1])
        
        self.delta_prev_read = np.vstack((self.delta_prev_read,d_read))
                                         
        self.delta_prev_read_weight = np.vstack((self.delta_prev_read_weight,d_prev_read_weight))
       
        
        delta_read_weight_dict, d_controller_read = self.ReadHeadParamGen.back_pass(d_read_params, 
                                                                self.controller_time[-1],self.read_weight_dict)
                                         
        
        for key in self.delta_read_weights.keys():
            self.delta_read_weights[key] += delta_read_weight_dict[key]
        
        #Backprop write head
        d_M_write, d_prev_write_weight, d_write_params = self.WriteHead.back_pass(
            d_M_read+self.d_Memory[-1], self.delta_prev_write_weight[-1], self.Memory[-1], 
            self.prev_write_weight[-1],self.Write_Parameters[-1])
        
        self.delta_prev_write_weight = np.vstack((self.delta_prev_write_weight,d_prev_write_weight))
        
        self.d_Memory = np.append(self.d_Memory,d_M_write[np.newaxis,:,:],axis=0)
        
        delta_write_weight_dict, d_controller_write = self.WriteHeadParamGen.back_pass(
            d_write_params, self.controller_time[-1],self.write_weight_dict)
                 
        for key in self.delta_write_weights.keys():
            self.delta_write_weights[key] += delta_write_weight_dict[key]
        
        self.delta_controller_read = np.vstack((self.delta_controller_read,d_controller_read))
        self.delta_controller_write = np.vstack((self.delta_controller_write,d_controller_write))
        
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
        
        return d_controller
    
    def weight_update(self):
        
        '''
        To update read_weight_dict, write_weight_dict, w_r_h, b_r_h using accumulated deltas
        
        Initial run will be without RMSProp or Momentum
        
        '''
        
        self.w_r_h += self.learning_rate*self.d_w_r_h
        self.b_r_h += self.learning_rate*self.d_b_r_h
        
        for key in self.read_weight_dict.keys():
            self.read_weight_dict[key] += self.learning_rate*self.delta_read_weights[key]
        
        for key in self.write_weight_dict.keys():
            self.write_weight_dict[key] += self.learning_rate*self.delta_write_weights[key]
        
        return None
    
    def reset(self, N=128, M=20, H=100):
        self.Memory = np.zeros((1,N,M))+0.1
        self.d_Memory = np.zeros((1,N,M))
        
        self.d_w_r_h = np.zeros((H,M))
        self.d_b_r_h = np.zeros(H)
        
        self.prev_read = np.zeros((1,M))
        self.prev_write_weight = np.zeros((1,N))
        self.prev_read_weight = np.zeros((1,N))
        
        self.Read_Parameters = []
        self.Write_Parameters = []
     
        self.delta_prev_read = np.zeros((1,M))
        self.delta_prev_write_weight = np.zeros((1,N))
        self.delta_prev_read_weight = np.zeros((1,N))
        
        self.delta_controller_write = np.zeros((1,H))
        self.delta_controller_read = np.zeros((1,H))
        
        self.delta_read_weights = {
            'w_r_k' : 0,
            'w_r_beta': 0,
            'w_r_g': 0,
            'w_r_s': 0,
            'w_r_gamma': 0,
            'b_r_k' : 0,
            'b_r_beta': 0,
            'b_r_g': 0,
            'b_r_s': 0,
            'b_r_gamma': 0,
        }

        self.delta_write_weights = {
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

