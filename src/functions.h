#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cstdint>
#include <eigen3/Eigen/Dense>

uint32_t swap_endianness(uint32_t input);
std::vector<Eigen::VectorXd> read_images(std::string path, int numLabels);
std::vector<int> read_labels(std::string path, int numLabels);

#endif // FUNCTIONS_H
