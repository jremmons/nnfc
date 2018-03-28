#include <cassert>
#include <iostream>

#include "noop.hh"
#include "array3d.hh"

Noop::Noop() {}
Noop::~Noop() {}

int Noop::encode(int x) {

    float* _data = new float[100000];
    Array3D<float> data(_data, 10, 100, 100);

    size_t count = 0;
    for(int i = 0; i < 10; i++){
        for(int j = 0; j < 100; j++){
            for(int k = 0; k < 100; k++){

                data.set((float)count, i, j, k);
                count += 1;
            }
        }
    }
 
    for(size_t i = 0; i < 100000; i++){
        if(i % 1000 == 0){
            std::cerr << _data[i] << " " << (float)i << std::endl;
        }


        if(_data[i] != (float)i){
            std::cerr << _data[i] << " " << (float)i << "\n";
        }
        assert(_data[i] == (float)i);
    }
    
    
    return x;

}

int Noop::decode(int x) { return x; }
