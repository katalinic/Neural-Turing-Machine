
import numpy as np
import NTM
import Util
import matplotlib.pyplot as plt


I = 10
O = 10
cntlr = 100

# Xavier weight initialisation
w_I = np.random.randn(cntlr,I)/np.sqrt(I+cntlr)
w_O = np.random.randn(O,cntlr)/np.sqrt(O+cntlr)
b_I = np.random.randn(cntlr)/np.sqrt(cntlr)
b_O = np.random.randn(O)/np.sqrt(O)

learning_rate = -1e-4

NTMach = NTM.NTM()

#single pass:

#limit to sequence length 10 initially

loss_seq = []

for q in range(10000):

    loss = 0
    
    sequence_length = np.random.randint(1,11)

    input_sequence, output_sequence = Util.Copy_seq_gen(vector_size=8, seq_length=sequence_length)

    total_d_w_I = 0
    total_d_b_I = 0
    total_d_w_O = 0
    total_d_b_O = 0

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

        total_d_w_O += d_w_O
        total_d_b_O += d_b_O

        d_cntlr_input_wbias = NTMach.back_pass(d_cntlr_out)

        d_cntlr_input, d_b_I = Util.Add.back_pass(d_cntlr_input_wbias)

        d_w_I, _ = Util.MatVecMul.back_pass(d_cntlr_input, w_I, input_sequence[:,k])

        total_d_w_I += d_w_I
        total_d_b_I += d_b_I

    output_test = output_test[:,1:]
    
    #Perform parameter updates
    w_I += learning_rate*total_d_w_I
    w_O += learning_rate*total_d_w_O
    b_I += learning_rate*total_d_b_I
    b_O += learning_rate*total_d_b_O
    NTMach.weight_update()

    loss_seq.append(loss)
   
