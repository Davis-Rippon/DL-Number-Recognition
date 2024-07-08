#include "functions.h"
#include <fstream>
#include <iostream>

uint32_t swap_endianness(uint32_t input) {
	return (input >> 24 |
			(input >> 8 & 0xff00) |
			(input << 8 & 0xff0000) | 
			input << 24);
}


std::vector<Eigen::VectorXd> read_images(std::string path, int numImages) {
	std::ifstream file(path);
	std::vector<Eigen::VectorXd> output(numImages);

	if (file.is_open()) {
		/* Start by skipping first 12 bytes of the file since we know the numnber of images, etc. */
		file.ignore(16);

		for (int i = 0; i < numImages; i++) {
			Eigen::VectorXd image(784);

			for (int j = 0; j < 784; j++) {
				char ch;
 				file.read(&ch, 1);
				image[j] = ch;
			}
			output[i] = image;
		}

	} else {
		throw std::runtime_error("Could not open MNIST database");
	}
	return output;
}

std::vector<int> read_labels(std::string path, int numLabels) {

	std::ifstream file(path);
	std::vector<int> output(numLabels);

	if (file.is_open()) {
		file.ignore(8);

		for (int i = 0; i < numLabels; i++) {
			char ch;
			file.read(&ch, 1);
			output[i] = ch;

			int num = int(uint8_t(ch));
			std::cout << num << std::endl;
		}

	} else {
		throw std::runtime_error("Could not open MNIST database");
	}
	return output;
}
