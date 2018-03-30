
int noop_encode_forward(THFloatTensor *input, THByteTensor *output);
int noop_encode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);

int noop_decode_forward(THByteTensor *input, THFloatTensor *output);
int noop_decode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);
