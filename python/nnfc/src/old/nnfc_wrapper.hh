
int nnfc_encode_forward(THFloatTensor *input, THByteTensor *output);
int nnfc_encode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);

int nnfc_decode_forward(THByteTensor *input, THFloatTensor *output);
int nnfc_decode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);
