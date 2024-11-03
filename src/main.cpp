#include "functions.h"
#include "NetworkMLP.h"
#include <iostream>
#include <iomanip>

int main() {
	std::string imageDatabase = "data/train-images-idx3-ubyte";
	std::string labelDatabase = "data/train-labels-idx1-ubyte";

	std::cout << "Reading Labels..." << std::flush;
	std::vector<int> labels = read_labels(labelDatabase, 10000);

	std::cout << " Done \n\nReading Images..." << std::flush;
	std::vector<Eigen::Vector<uint8_t, 784>> images = read_images(imageDatabase, 60000);

	std::cout << " Done \n\nInitialising Network..." << std::flush;
	NetworkMLP networkMLP = NetworkMLP();

	float sigmoid_val = 12.123;
	std::cout << " Done \n\nSigmoid Test..." << std::flush << std::endl;

	for (int i = 0; i < 10; ++i) {
		std::cout << "Fast Sigmoid (input: " << sigmoid_val + i << ") " << fast_sigmoid(sigmoid_val + i) << std::setprecision(5) << "\n" << std::flush;
	}

	std::cout << "Done \n\nBack Propagation..." << std::endl;
	networkMLP.back_prop(images, labels, 0.05, 10);

	return 0;
}
