#include <iostream>

#include "noop.hh"

int main(int argc, char* argv[]){

    std::cout << argc << " " << argv[0] << "\n";
    std::cout << "hello world\n";

    Noop n;

    std::cout << "42 == " << n.encode(42) << "\n";
    std::cout << "777 == " << n.decode(777) << "\n";
    
    return 0;
}
