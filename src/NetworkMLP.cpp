#include "NetworkMLP.h"
#include "functions.h"
#include <iostream>
 
NetworkMLP::NetworkMLP() {
	weightsL1 = Eigen::Vector<float, 12544>::Random();
	biasesL1 = Eigen::Vector<float, 16>::Random();

	weightsL2 = Eigen::Vector<float, 256>::Random();
	biasesL2 = Eigen::Vector<float, 16>::Random();

	weightsL3 = Eigen::Vector<float, 160>::Random();
	biasesL3 = Eigen::Vector<float, 10>::Random();
}

float NetworkMLP::loss_function(Eigen::Vector<float, 10> output, int expected) {
	float loss = 0;
	for (int i = 0; i < 10; ++i) {
		if (i == expected) loss += (output[i] - 1)*(output[i] - 1);
		else loss += output[i]*output[i];
	}
	return loss;
}

void NetworkMLP::back_prop(
		std::vector<Eigen::Vector<float,784>> input,
		std::vector<int> expectedOutput,
		float learningRate,
		int tolerance,
		int epochs
		) {

	int nInputs = input.size();
	/*Adjusts the different weights and biases in hidden layers 1-3 according to input images and output labels*/
	for (int e = 0; e < epochs; ++e) {
		float total_loss = 0;

		for (int i = 0; i < nInputs; ++i) {
			std::vector<Eigen::Vector<float, 10>> layers = forward_pass(input[i]);

			float loss = loss_function(layers[2], expectedOutput[i]);
			total_loss += loss;

			/* Output Error */
			Eigen::Vector<float, 10> D1;

			for (int j = 0; j < 10; ++j) {
				int exp = (j == expectedOutput[i]) ? 1 : 0;
				D1[j] = layers[2][j] - exp;
			}

			/* Hidden Layer 2 error */
			Eigen::Vector<float, 16> D2;
			for (int j = 0; j < 16; ++j) {
				float error = 0;
				for (int k = 0; k < 10; ++k) {
					error += weightsL2[k]*D1[k];
				}
				D2[j] = error*tanh_derivative(layers[1][j]);
			}


			/* Hidden Layer 1 error */
			Eigen::Vector<float, 784> D3;
			for (int j = 0; j < 784; ++j) {
				float error = 0;
				for (int k = 0; k < 10; ++k) {
					error += weightsL1[k]*D2[k];
				}
				D3[j] = error*tanh_derivative(layers[0][j]);
			}

			/* Gradient Vector for Layer 2 */
			Eigen::Vector<float, 160> weightGradL2;
			for (int j = 0; j < 10; ++j) {
				for (int k = 0; k < 16; ++k) {
					weightGradL2[j*784 + k] = D1[j] * layers[1][k];
				}
			}


			/* Gradient Vector for Layer 1 */
			Eigen::Vector<float, 12544> weightGradL1;
			for (int j = 0; j < 16; ++j) {
				for (int k = 0; k < 784; ++k) {
					weightGradL1[j*784 + k] = D2[j] * input[i][k];
				}
			}


		}
	}
}

std::vector<Eigen::Vector<float, 10>> NetworkMLP::forward_pass(Eigen::Vector<float, 784> image) {

	/*Does one forward pass on network, returning output layer*/
	Eigen::Vector<float, 16> Z1;

	for (int i = 0; i < 16; ++i) {
		float sum = 0;
		for (int j = 0; j < 784; ++j) {
			sum += ((unsigned int) image[j])*weightsL1[j + i*784];
		}
		sum += biasesL1[i];
		float value = tanh_squish(sum);
		Z1[i] = value;
	}

	Eigen::Vector<float, 16> Z2;
	for (int i = 0; i < 16; ++i) {
		float sum = 0;
		for (int j = 0; j < 16; ++j) {
			sum += Z1[j]*weightsL2[j + i*16];
		}
		sum += biasesL2[i];
		float value = tanh_squish(sum);
		Z2[i] = value;
	}

	Eigen::Vector<float, 10> Z3;
	for (int i = 0; i < 10; ++i) {
			float sum = 0;
			for (int j = 0; j < 16; ++j) {
				sum += Z2[j]*weightsL3[j + i*16];
			}
			sum += biasesL3[i];
			float value = tanh_squish(sum);
			Z3[i] = value;
		}

	/*for (int i = 0; i < 10; ++i) {*/
	/*	std::cout << Z3[i] << " " << Z2[i] << " " << Z1[i] << " " << std::endl;*/
	/*}*/

	return	{Z1, Z2, Z3};
}

float test_accuracy(std::vector<Eigen::Vector<uint8_t, 784>> input, std::vector<int> expectedOutput) {
	/*Tests the accuracy of the model using the test labels*/
	return 1.0;
}
