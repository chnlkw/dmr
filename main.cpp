#include <iostream>
#include <vector>

#include "dmr.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    int nworkers = 4;
    std::vector<DMR> dmr;
    for (size_t i = 0; i < nworkers; i++)
        dmr.emplace_back(nworkers, i);

    return 0;
}