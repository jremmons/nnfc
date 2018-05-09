#include <iostream>

#include "tensor.hh"
#include "nnfc_CXXAPI.hh"

int main(int, char* [])
{

    auto available_contexts = nnfc::cxxapi::get_available_encoders();
    for(size_t i = 0; i < available_contexts.size(); i++) {
        std::cout << available_contexts[i] << "\n";
    }
        
    
    
    return 0;
}
