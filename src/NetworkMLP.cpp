#include "NetworkMLP.h"
#include <iostream>
 
NetworkMLP::NetworkMLP() {
	weightsL1 = Eigen::Vector<uint8_t, 12544>::Random();
	biasesL1 = Eigen::Vector<uint8_t, 16>::Random();

	weightsL2 = Eigen::Vector<uint8_t, 256>::Random();
	biasesL2 = Eigen::Vector<uint8_t, 16>::Random();

	weightsL3 = Eigen::Vector<uint8_t, 160>::Random();
	biasesL3 = Eigen::Vector<uint8_t, 10>::Random();
}

void NetworkMLP::back_prop(std::vector<Eigen::Vector<uint8_t,784>> input, std::vector<int> expectedOutput) {
	/*Adjusts the different weights and biases in hidden layers 1-3 according to input images and output labels*/
}


int NetworkMLP::forward_pass(Eigen::Vector<uint8_t, 784> image) {
	return 0;
}

float test_accuracy(std::vector<Eigen::Vector<uint8_t, 784>> input, std::vector<int> expectedOutput) {
	return 1.0;
}
