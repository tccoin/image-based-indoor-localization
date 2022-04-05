//
// Created by lepet on 4/3/2022.
//

#include <vector>
#include <iostream>
#include <chrono>

#include "lib.h"

int main() {
    std::vector<double> data(128 * 1000000);

    for (int i = 0; i < data.size(); ++i) data[i] = (double) i;

    init_data(data.data(), data.size());

    auto total = std::chrono::high_resolution_clock::duration(0);

    for (int i = 0; i < 100; ++i) {
        std::vector<double> input(128, 1000 * i);

        auto tic = std::chrono::high_resolution_clock::now();
        find_nearest(input.data());
        auto toc = std::chrono::high_resolution_clock::now();

        total += toc - tic;
    }

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(total).count();

    return 0;
}