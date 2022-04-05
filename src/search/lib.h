//
// Created by lepet on 4/4/2022.
//

#ifndef SEARCH_SEARCH_H
#define SEARCH_SEARCH_H

#ifdef __linux__
#define __declspec(v)
#endif

extern "C" {
__declspec(dllexport) void init_data(const double* data, size_t size);
__declspec(dllexport) int find_nearest(const double* input);
};

#endif //SEARCH_SEARCH_H
