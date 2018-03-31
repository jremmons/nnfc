
int device_to_host_copy(THFloatTensor *dest, THCudaTensor *src);

int alloc_pinned_tensor_float(THFloatTensor *tensor);
int free_pinned_tensor_float(THFloatTensor *tensor);

int alloc_pinned_tensor_byte(THByteTensor *tensor);
int free_pinned_tensor_byte(THByteTensor *tensor);
