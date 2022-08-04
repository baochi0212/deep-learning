#in the decoder:
q != k, v 
here it is:
k, v is batch x [:curr_time] x feat
q is batch x 1(curr_time) x feat -> get the attention weight at each time step (1 x feature @ feature x n -> 1 x n (weight) @ 
n x feature -> 1 x feature) -> the attention weighted value of queried postion !!!!

#in the encoder: q, k, v same 
decoder: q at each time step is different from k, v 
add norm: cat the queried value and prev pos -> like the history attention 