#include "functions.h"
#include <iostream>
#include <iomanip>

int main() {
	std::string path = "data/train-images-idx3-ubyte";
	read_images(path, 60000);

	return 0;
}
