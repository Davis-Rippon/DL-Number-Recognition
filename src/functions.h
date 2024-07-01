#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cstdint>
#include <eigen3/Eigen/Dense>

uint32_t swap_endianness(uint32_t input);

std::vector<Eigen::MatrixXd> read_images(std::string path);

#endif // FUNCTIONS_H
