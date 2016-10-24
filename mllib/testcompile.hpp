#pragma once
#include <random>
#include <string>
#include <vector>

#include "core/engine.hpp"

class PIObject {
   public:
    typedef int KeyT;
    int key;

    explicit PIObject(KeyT key);

    const int& id() const;
};

void pi();
