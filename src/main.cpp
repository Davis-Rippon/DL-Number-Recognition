#include "functions.h"
#include <iostream>
#include <iomanip>

int main() {
	int val = (int) 0x12345678;

	int reverse = swap_endianness(val);

	std::cout << std::hex << reverse << std::endl;
	std::cout << std::hex << val << std::endl;

	return 0;
}
