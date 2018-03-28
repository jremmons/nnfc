#pragma once

class Noop {
public:
    Noop();
    ~Noop();
    
    int encode(int x);
    int decode(int x);
};
