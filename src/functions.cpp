#include "functions.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

uint32_t swap_endianness(uint32_t input) {
	return (input >> 24 |
			(input >> 8 & 0xff00) |
			(input << 8 & 0xff0000) | 
			input << 24);
}

float fast_sigmoid(float x) {
	return 1/(1 + abs(x));
}

float sigmoid(float x) {
	if (x < -20) std::cout << "Overflow: " << x << std::endl;
    return 1.0f / (1.0f + std::exp(-x));
}

float tanh_squish(float x) {
	return std::tanh(x);
}

void show_progress(int current, int max, int barLen=40) {
	std::cout << '\r';
	std::cout << '[';

	if (current == 0) {
		for (int i = 0; i < barLen; ++i) {
			std::cout << '-';
		}

	} else {

		float progIdx = current / (max/barLen);

		for (int i = 0; i < progIdx; ++i) {
			std::cout << '#';
		}

		for (int i = progIdx + 1; i < barLen - 1; ++i) {
			std::cout << '-';
		}

	}

	std::cout << "] " << std::flush;
	std::cout << (current + 1);
}

std::vector<Eigen::Vector<float, 784>> read_images(std::string path, int numImages) {
	std::ifstream file(path);
	std::vector<Eigen::Vector<float, 784>> output(numImages);

	if (file.is_open()) {
		/* Start by skipping first 12 bytes of the file since we know the numnber of images, etc. */
		file.ignore(16);

		for (int i = 0; i < numImages; i++) {
			/*show_progress(i, 60000,	50);*/

			Eigen::Vector<float, 784> image;

			for (int j = 0; j < 784; j++) {
				char ch;
 				file.read(&ch, 1);
				image[j] = fast_sigmoid((float) ch);
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
			/*show_progress(i, numLabels);*/
			char ch;
			file.read(&ch, 1);
			output[i] = ch;
		}

	} else {
		throw std::runtime_error("Could not open MNIST database");
	}
	return output;
}


void show_number(std::vector<Eigen::Vector<float, 784>> images, int index) {
	for (int j = 0; j < 28; ++j) {
		for (int i = 0; i < 28; ++i) {
			std::cout << std::setw(2) << (int)images[index][28 * j + i];
		}
		std::cout << std::endl;
	}
}

