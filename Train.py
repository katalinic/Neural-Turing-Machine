
import numpy as np
import NTM
import Util
import matplotlib.pyplot as plt


I = 10
O = 10
cntlr = 100
decay_param = 0.95

# Xavier weight initialisation
w_I = np.random.randn(cntlr,I)/np.sqrt(I+cntlr)
w_O = np.random.randn(O,cntlr)/np.sqrt(O+cntlr)
b_I = np.random.randn(cntlr)/np.sqrt(cntlr)
b_O = np.random.randn(O)/np.sqrt(O)

learning_rate = 1e-4

NTMach = NTM.NTM()

#single pass:

#limit to sequence length 10 initially

loss_seq = []

s_w_I, s_w_O, s_b_I, s_b_O = np.zeros_like(w_I), np.zeros_like(w_O), np.zeros_like(b_I), np.zeros_like(b_O)
rms_w_I, rms_w_O, rms_b_I, rms_b_O = np.zeros_like(w_I), np.zeros_like(w_O), np.zeros_like(b_I), np.zeros_like(b_O)

for q in range(1):

    loss = 0
    
    sequence_length = np.random.randint(1,11)

    input_sequence, output_sequence = Util.Copy_seq_gen(vector_size=8, seq_length=sequence_length)

    NTMach.reset()
    
    output_test = np.zeros((O,1))
    
    #Forward pass
    for k in range(input_sequence.shape[1]):

        sig_act = Util.Sigmoid()

        cntlr_input = Util.MatVecMul.fwd_pass(w_I,input_sequence[:,k])
        cntlr_input_wbias = Util.Add.fwd_pass(cntlr_input, b_I)

        cntlr_out = NTMach.fwd_pass(cntlr_input_wbias)

        y_input = Util.MatVecMul.fwd_pass(w_O,cntlr_out)
        y_input_wbias = Util.Add.fwd_pass(y_input, b_O)
        y_sigmoid = sig_act.fwd_pass(y_input_wbias)
        
        output_test = np.hstack((output_test,y_sigmoid[:,np.newaxis]))
        
        loss += Util.CrossEntropyLossSigmoid.fwd_pass(y_sigmoid, output_sequence[:,k])

        d_y_sigmoid = Util.CrossEntropyLossSigmoid.back_pass(y_sigmoid, output_sequence[:,k])
        
        d_y_input_wbias = sig_act.back_pass(d_y_sigmoid)
        
        d_y_input, d_b_O = Util.Add.back_pass(d_y_input_wbias)

        d_w_O, d_cntlr_out = Util.MatVecMul.back_pass(d_y_input, w_O, cntlr_out)

        s_w_O += d_w_O
        s_b_O += d_b_O

        d_cntlr_input_wbias = NTMach.back_pass(d_cntlr_out)

        d_cntlr_input, d_b_I = Util.Add.back_pass(d_cntlr_input_wbias)

        d_w_I, _ = Util.MatVecMul.back_pass(d_cntlr_input, w_I, input_sequence[:,k])

        s_w_I += d_w_I
        s_b_I += d_b_I

    output_test = output_test[:,1:]
    loss_seq.append(loss)
    
    s_w_I = Util.Clip(s_w_I)
    s_w_O = Util.Clip(s_w_O)
    s_b_I = Util.Clip(s_b_I)
    s_b_O = Util.Clip(s_b_O)
    
    #Perform parameter updates    
    rms_w_I = decay_param*rms_w_I + (1-decay_param)*s_w_I**2
    rms_w_O = decay_param*rms_w_O + (1-decay_param)*s_w_O**2
    rms_b_I = decay_param*rms_b_I + (1-decay_param)*s_b_I**2
    rms_b_O = decay_param*rms_b_O + (1-decay_param)*s_b_O**2
    
    w_I -= learning_rate*s_w_I/(np.sqrt(rms_w_I)+1e-4)
    w_O -= learning_rate*s_w_O/(np.sqrt(rms_w_O)+1e-4)
    b_I -= learning_rate*s_b_I/(np.sqrt(rms_b_I)+1e-4)
    b_O -= learning_rate*s_b_O/(np.sqrt(rms_b_O)+1e-4)
    
    NTMach.weight_update()

    s_w_I, s_w_O, s_b_I, s_b_O = np.zeros_like(w_I), np.zeros_like(w_O), np.zeros_like(b_I), np.zeros_like(b_O)
    
    
