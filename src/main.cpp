#include "functions.h"
#include "NetworkMLP.h"
#include <iostream>
#include <iomanip>

int main() {
	std::string imageDatabase = "data/train-images-idx3-ubyte";
	std::string labelDatabase = "data/train-labels-idx1-ubyte";

	std::cout << "Reading Labels...";
	std::vector<int> labels = read_labels(labelDatabase, 10000);

	std::cout << " Done \nReading Images...";
	std::vector<Eigen::Vector<uint8_t, 784>> images = read_images(imageDatabase, 60000);

	std::cout << " Done \nInitialising Network...";
	NetworkMLP networkMLP = NetworkMLP();

	std::cout << " Done \nBack Propagation" << std::endl;

	return 0;
}
