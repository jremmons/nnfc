
int encode_forward(THFloatTensor *input, THByteTensor *output);
int encode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);

int decode_forward(THByteTensor *input, THFloatTensor *output);
int decode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);
