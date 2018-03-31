#include <TH/TH.h>
#include <THC/THC.h>
#include <THC/THCGeneral.h>
#include <THC/THCTensor.h>

#include <sys/mman.h>

#include <chrono>
#include <cstring>
#include <iostream>

#include "common.hh"

extern THCState *state;

extern "C" int alloc_pinned_tensor_float(THFloatTensor *tensor){

    void *pinned_data;
    size_t size = THFloatTensor_nElement(tensor);
    THCudaCheck(cudaMallocHost((void**)&pinned_data, size*sizeof(float) + sizeof(void*)));

    *reinterpret_cast<void**>(static_cast<void*>(pinned_data) + size) = tensor->storage->data; // store the old array as a ptr at the end of the new array
    tensor->storage->data = static_cast<float*>(pinned_data);

    return _TORCH_SUCCESS;
}

extern "C" int free_pinned_tensor_float(THFloatTensor *tensor){

    size_t size = THFloatTensor_nElement(tensor);
    float *ptr = *reinterpret_cast<float**>(static_cast<void*>(tensor->storage->data) + size*sizeof(float));
        
    THCudaCheck(cudaFreeHost(tensor->storage->data));

    tensor->storage->data = ptr;

    return _TORCH_SUCCESS;
}


extern "C" int alloc_pinned_tensor_byte(THByteTensor *tensor){

    size_t size = THByteTensor_nElement(tensor);
    void *pinned_data;
    THCudaCheck(cudaMallocHost((void**)&pinned_data, size*sizeof(uint8_t) + sizeof(void*)));

    *reinterpret_cast<void**>(static_cast<void*>(pinned_data) + size) = tensor->storage->data; // store the old array as a ptr at the end of the new array
    tensor->storage->data = static_cast<uint8_t*>(pinned_data);

    return _TORCH_SUCCESS;
}


extern "C" int free_pinned_tensor_byte(THByteTensor *tensor){

    size_t size = THByteTensor_nElement(tensor);
    uint8_t *ptr = *reinterpret_cast<uint8_t**>(static_cast<void*>(tensor->storage->data) + size*sizeof(uint8_t));
        
    THCudaCheck(cudaFreeHost(tensor->storage->data));

    tensor->storage->data = ptr;

    return _TORCH_SUCCESS;
}


extern "C" int device_to_host_copy(THFloatTensor *dest, THCudaTensor *src){

    size_t size_src = THCudaTensor_nElement(state, src);
    size_t size_dest = THFloatTensor_nElement(dest);
    THArgCheck(size_dest == size_src, 2, "sizes do not match"); 

    THCudaCheck(cudaMemcpy(dest->storage->data + dest->storageOffset, src->storage->data + src->storageOffset, size_dest*sizeof(float), cudaMemcpyDeviceToHost));
    THCudaTensor_free(state, src);
    
    return _TORCH_SUCCESS;
}

