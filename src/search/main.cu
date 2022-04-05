//
// Created by lepet on 4/3/2022.
//

#include <vector>
#include <iostream>

#include "lib.h"

int main() {
    std::vector<double> data(128 * 4000);
    std::vector<double> input(128, 1600);

    for (int i = 0; i < data.size(); ++i) data[i] = (double) i;

    init_data(data.data(), data.size());
    std::cout << find_nearest(input.data());

    return 0;
}