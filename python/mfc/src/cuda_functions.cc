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
//extern THAllocator THDefaultAllocator;

// static float *data_ = NULL; 
// static size_t data_size_ = 0;

extern "C" int alloc_pinned_tensor_float(THFloatTensor *tensor){

    size_t size = THFloatTensor_nElement(tensor);
    std::cerr << "tensor size: " << size << std::endl;
    void *pinned_data;
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
    //THFloatTensor_free(tensor);

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
    //THByteTensor_free(tensor);

    return _TORCH_SUCCESS;
}


extern "C" int device_to_host_copy(THFloatTensor *dest, THCudaTensor *src){

    size_t size_src = THCudaTensor_nElement(state, src);
    size_t size_dest = THFloatTensor_nElement(dest);
    //std::cerr << size_src << " " << size_dest << std::endl;
    THArgCheck(size_dest == size_src, 2, "sizes do not match"); 
    
    // if(data_ == NULL){
    //     cudaMallocHost((void**)&data_, size*sizeof(float) + sizeof(float*), cudaHostAllocPortable);
    //     data_size_ = size*sizeof(float) + sizeof(float*);
    //     std::cerr << "alloc!" << std::endl;
    // }

    // if(data_size_ != size*sizeof(float) + sizeof(float*)){
    //     cudaFreeHost(data_);

    //     cudaMallocHost((void**)&data_, size*sizeof(float) + sizeof(float*), cudaHostAllocPortable);
    //     data_size_ = size*sizeof(float) + sizeof(float*);
    //     std::cerr << "realloc!" << std::endl;
    // }
    
    //void *pinned_data = data_;
    //std::cerr << "set_ptr!\n";

    THCudaCheck(cudaMemcpy(dest->storage->data + dest->storageOffset, src->storage->data + src->storageOffset, size_dest*sizeof(float), cudaMemcpyDeviceToHost));

    // auto memcpy_t1 = std::chrono::high_resolution_clock::now();
    // *reinterpret_cast<float**>(static_cast<void*>(pinned_data) + size*sizeof(float)) = dest->storage->data; // store the old array as a ptr at the end of the new array
    // dest->storage->data = (float*) pinned_data;
    // auto memcpy_t2 = std::chrono::high_resolution_clock::now();

    THCudaTensor_free(state, src);
    
    return _TORCH_SUCCESS;
}

