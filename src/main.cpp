#include "functions.h"
#include <iostream>
#include <iomanip>

int main() {
	std::string imageDatabase = "data/train-images-idx3-ubyte";
	std::string labelDatabase = "data/train-labels-idx1-ubyte";

	std::vector<int> labels = read_labels(labelDatabase, 10000);
	std::vector<Eigen::VectorXd> images = read_images(imageDatabase, 60000);

	return 0;
}
