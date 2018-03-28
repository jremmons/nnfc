#include "noop.hh"

Noop::Noop() {}

Noop::~Noop() {}

int Noop::encode(int x) { return x; }

int Noop::decode(int x) { return x; }
