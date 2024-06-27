#include "functions.h"
#include <algorithm>
#include <iostream>

float ReLU(float x) {
	/*Rectifier function that converts summed activation to output*/
	return (x < 0) ? 0 : x;
}

