import numpy as np
import NTM2
import Util

def train(sequence_length=5, max_eps=50000, convergence_criterion=1e-3):
    
    print ("Commencing training")
    
    I = 10
    O = 10
    cntlr = 100
    decay_param = 0.95
    momentum = 0.9
    learning_rate = 1e-4

    w_I = np.random.randn(cntlr,I)/np.sqrt(I+cntlr)
    w_O = np.random.randn(O,cntlr)/np.sqrt(O+cntlr)
    b_I = np.zeros(cntlr)
    b_O = np.zeros(O)

    NTMach = NTM2.NTM()

    loss_seq = []

    rms_w_I, rms_w_O, rms_b_I, rms_b_O = np.zeros_like(w_I), np.zeros_like(w_O), np.zeros_like(b_I), np.zeros_like(b_O)
    moment_w_I, moment_w_O, moment_b_I, moment_b_O = np.zeros_like(w_I), np.zeros_like(w_O), np.zeros_like(b_I), np.zeros_like(b_O)
    rms_grad_w_I, rms_grad_w_O, rms_grad_b_I, rms_grad_b_O =     np.zeros_like(w_I), np.zeros_like(w_O), np.zeros_like(b_I), np.zeros_like(b_O)

    s_w_I, s_w_O, s_b_I, s_b_O = np.zeros_like(w_I), np.zeros_like(w_O), np.zeros_like(b_I), np.zeros_like(b_O)

    for ep in range(max_eps):

        input_sequence, output_sequence = Util.Copy_seq_gen(vector_size=8, seq_length=sequence_length)

        controller_inst = []
        output_activation_inst = []
        y_sigmoid = []
        z = []

        #Forward pass
        for k in range(input_sequence.shape[1]):

            output_activation_inst.append(Util.Sigmoid())

            cntlr_input = Util.MatVecMul.fwd_pass(w_I,input_sequence[:,k])
            cntlr_input_wbias = Util.Add.fwd_pass(cntlr_input, b_I)

            controller_inst.append(NTMach.fwd_pass(cntlr_input_wbias))

            z.append(np.tanh(controller_inst[k]))

            y_input = Util.MatVecMul.fwd_pass(w_O,z[k])

            y_input_wbias = Util.Add.fwd_pass(y_input, b_O)

            y_sigmoid.append(output_activation_inst[k].fwd_pass(y_input_wbias))

        loss = 0

        for q in reversed(range(output_sequence.shape[1])):

            loss += Util.CrossEntropyLossSigmoid.fwd_pass(y_sigmoid[q], output_sequence[:,q])

            d_y_sigmoid = Util.CrossEntropyLossSigmoid.back_pass(y_sigmoid[q], output_sequence[:,q])

            d_y_input_wbias = output_activation_inst[q].back_pass(d_y_sigmoid)

            d_y_input, d_b_O = Util.Add.back_pass(d_y_input_wbias)

            d_w_O, d_z = Util.MatVecMul.back_pass(d_y_input, w_O, z[q])

            s_w_O += d_w_O
            s_b_O += d_b_O

            d_cntlr_out = d_z*(1-z[q]**2)

            d_cntlr_input_wbias = NTMach.back_pass(d_cntlr_out)

            d_cntlr_input, d_b_I = Util.Add.back_pass(d_cntlr_input_wbias)

            d_w_I, _ = Util.MatVecMul.back_pass(d_cntlr_input, w_I, input_sequence[:,q])

            s_w_I += d_w_I
            s_b_I += d_b_I
        
        if loss==0:
            print ("Error: Numerical instability encountered.")
            return None
        
        if loss<convergence_criterion:
            print ("Successfully converged at: Sequence number: ", ep, "Loss: ", loss)
            return None
        
        if ep%1000==0:
            print ("Training progress: Sequence number: ", ep, "Loss: ", loss)
        
        s_w_I = Util.Clip(s_w_I)
        s_w_O = Util.Clip(s_w_O)
        s_b_I = Util.Clip(s_b_I)
        s_b_O = Util.Clip(s_b_O)

        #Perform parameter updates    
        rms_w_I = decay_param*rms_w_I + (1-decay_param)*s_w_I**2
        rms_w_O = decay_param*rms_w_O + (1-decay_param)*s_w_O**2
        rms_b_I = decay_param*rms_b_I + (1-decay_param)*s_b_I**2
        rms_b_O = decay_param*rms_b_O + (1-decay_param)*s_b_O**2

        moment_w_I = decay_param*moment_w_I + (1-decay_param)*s_w_I
        moment_w_O = decay_param*moment_w_O + (1-decay_param)*s_w_O
        moment_b_I = decay_param*moment_b_I + (1-decay_param)*s_b_I
        moment_b_O = decay_param*moment_b_O + (1-decay_param)*s_b_O

        rms_grad_w_I = momentum*rms_grad_w_I - learning_rate*s_w_I/np.sqrt(rms_w_I-moment_w_I**2+1e-4)
        rms_grad_w_O = momentum*rms_grad_w_O - learning_rate*s_w_O/np.sqrt(rms_w_O-moment_w_O**2+1e-4)
        rms_grad_b_I = momentum*rms_grad_b_I - learning_rate*s_b_I/np.sqrt(rms_b_I-moment_b_I**2+1e-4)
        rms_grad_b_O = momentum*rms_grad_b_O - learning_rate*s_b_O/np.sqrt(rms_b_O-moment_b_O**2+1e-4)

        w_I += rms_grad_w_I
        w_O += rms_grad_w_O
        b_I += rms_grad_b_I
        b_O += rms_grad_b_O

        NTMach.weight_update()

        NTMach.reset()

        s_w_I, s_w_O, s_b_I, s_b_O = np.zeros_like(w_I), np.zeros_like(w_O), np.zeros_like(b_I), np.zeros_like(b_O) 
    
    print ("Convergence criterion not satisfied.")
    return None
    
if __name__=="__main__":
    train()

