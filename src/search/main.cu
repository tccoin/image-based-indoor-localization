//
// Created by lepet on 4/3/2022.
//

#include <vector>
#include <iostream>
#include <chrono>

#include "lib.h"

int main() {
    std::vector<double> data(128 * 1000000);

    for (int i = 0; i < data.size(); ++i) data[i] = (double) i / 128;

    init_data(data.data(), data.size());

    {
        std::vector<double> input(128, 100.5);
        std::cout << "Nearest: " << find_nearest(input.data()) << '\n';
    }

    auto total = std::chrono::high_resolution_clock::duration(0);

    for (int i = 0; i < 100; ++i) {
        std::vector<double> input(128, i);

        auto tic = std::chrono::high_resolution_clock::now();
        find_nearest(input.data());
        auto toc = std::chrono::high_resolution_clock::now();

        total += toc - tic;
    }

    std::cout << "Time for 100 searches: " << std::chrono::duration_cast<std::chrono::milliseconds>(total).count();

    cleanup();

    return 0;
}