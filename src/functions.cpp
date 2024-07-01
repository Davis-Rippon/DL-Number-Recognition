#include "functions.h"
#include <iostream>

uint32_t swap_endianness(uint32_t input) {
	return (input >> 24 |
			(input >> 8 & 0xff00) |
			(input << 8 & 0xff0000) | 
			input << 24);
}
