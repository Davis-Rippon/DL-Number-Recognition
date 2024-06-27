#include "functions.h"
#include <iostream>

int main () {
	std::cout << ReLU(1.1) << "\n";
	std::cout << ReLU(-1.0) << "\n";
	std::cout << ReLU(1.9) << "\n";

	return 0;
}
