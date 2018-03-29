#include <cassert>
#include <iostream>

#include "noop.hh"
#include "array3d.hh"

Noop::Noop() {}
Noop::~Noop() {}

Array3D<float> Noop::encode(Array3D<float> input) {

    return input;
}

Array3D<float> Noop::decode(Array3D<float> input) {

    return input;
}
