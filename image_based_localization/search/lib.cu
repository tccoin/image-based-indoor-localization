//
// Created by lepet on 4/4/2022.
//

#include <memory>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/zip_function.h>

#include "lib.h"

using thrust::device_vector;
using thrust::make_counting_iterator;
using thrust::make_transform_iterator;
using thrust::make_zip_iterator;
using thrust::make_permutation_iterator;

constexpr size_t vector_length = 128;

std::unique_ptr<device_vector<double>> dev_data(nullptr);

template<typename T>
struct key_functor {
    __host__ __device__
    constexpr T operator()(T index) {
        return index / T(vector_length);
    }
};

template<typename T>
struct input_functor {
    __host__ __device__
    constexpr T operator()(T index) {
        return index % T(vector_length);
    }
};

template<typename T>
struct diff_functor {
    template<class Tuple>
    __host__ __device__
    constexpr T operator()(Tuple t) {
        return thrust::get<0>(t) - thrust::get<1>(t);
    }
};

void init_data(const double* data, size_t size) {
    dev_data = std::make_unique<device_vector<double>>(size);
    copy(data, data + size, dev_data->begin());
}

int find_nearest(const double* input) {
    if (!dev_data) {
        std::cerr << "Error: Database not loaded. Please call init_data() first." << std::endl;
        return -1;
    }

    device_vector<double> dev_reduced_keys(dev_data->size() / 128);
    device_vector<double> dev_reduced_data(dev_data->size() / 128);

    device_vector<double> dev_input(vector_length);
    copy(input, input + vector_length, dev_input.begin());

    auto input_counter = make_counting_iterator(0);
    auto repeat_input_counter = make_transform_iterator(input_counter, input_functor<int>());
    auto input_iter = make_permutation_iterator(dev_input.begin(), repeat_input_counter);
    auto zip_iter = make_zip_iterator(input_iter, dev_data->begin());
    auto diff_iter = make_transform_iterator(zip_iter, diff_functor<double>());
    auto square_iter = make_transform_iterator(diff_iter, thrust::square<double>());

    auto key_counter_begin = make_counting_iterator(0);
    auto key_counter_end = key_counter_begin + dev_data->size();

    auto key_begin = make_transform_iterator(key_counter_begin, key_functor<int>());
    auto key_end = make_transform_iterator(key_counter_end, key_functor<int>());

    thrust::reduce_by_key(key_begin, key_end, square_iter, dev_reduced_keys.begin(), dev_reduced_data.begin());
    auto min = thrust::min_element(dev_reduced_data.begin(), dev_reduced_data.end());

    return min - dev_reduced_data.begin();
}

void cleanup() {
    dev_data = nullptr;
}
