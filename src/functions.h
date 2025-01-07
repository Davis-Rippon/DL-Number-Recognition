#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cstdint>
#include <eigen3/Eigen/Dense>

uint32_t swap_endianness(uint32_t input);
float fast_sigmoid(float x);
float sigmoid(float x);
float tanh_squish(float x);
float tanh_derivative(float x);
void show_progress(int current, int max, int barLen);
std::vector<Eigen::Vector<float, 784>> read_images(std::string path, int numLabels);
std::vector<int> read_labels(std::string path, int numLabels);
void show_number(std::vector<Eigen::Vector<float, 784>> image, int index);

#endif // FUNCTIONS_H
