#include "NetworkMLP.h"
#include "functions.h"
#include <iostream>
#include <fstream>
 
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
    std::vector<Eigen::Vector<float, 784>> input,
    std::vector<int> expectedOutput,
    float learningRate,
    float tolerance,
	float differenceThreshhold,
    int epochs
) {
    int nInputs = input.size();

	float previous = 0;

    for (int e = 0; e < epochs; ++e) {
        float total_loss = 0;

        for (int i = 0; i < nInputs; ++i) {
            ForwardPassContainer layers = forward_pass(input[i]);

            Eigen::Vector<float, 16> Z1 = layers.Z1;
            Eigen::Vector<float, 16> Z2 = layers.Z2;
            Eigen::Vector<float, 10> Z3 = layers.Z3;

            float loss = loss_function(Z3, expectedOutput[i]);
            total_loss += loss;

            Eigen::Vector<float, 10> D1;
            for (int j = 0; j < 10; ++j) {
                int exp = (j == expectedOutput[i]) ? 1 : 0; 
                D1[j] = Z3[j] - exp;
            }

            Eigen::Vector<float, 16> D2;
            for (int j = 0; j < 16; ++j) {
                float error = 0;
                for (int k = 0; k < 10; ++k) {
                    error += D1[k] * weightsL3[j + k * 16]; 
                }
                D2[j] = error * tanh_derivative(Z2[j]); 
            }

            Eigen::Vector<float, 16> D3;
            for (int j = 0; j < 16; ++j) {
                float error = 0;
                for (int k = 0; k < 16; ++k) {
                    error += D2[k] * weightsL2[j + k * 16];
                }
                D3[j] = error * tanh_derivative(Z1[j]);
            }

            for (int j = 0; j < 10; ++j) {
                for (int k = 0; k < 16; ++k) {
                    weightsL3[k + j * 16] -= learningRate * D1[j] * Z2[k];
                }
                biasesL3[j] -= learningRate * D1[j];
            }

            for (int j = 0; j < 16; ++j) {
                for (int k = 0; k < 16; ++k) {
                    weightsL2[k + j * 16] -= learningRate * D2[j] * Z1[k];
                }
                biasesL2[j] -= learningRate * D2[j];
            }

            for (int j = 0; j < 16; ++j) {
                for (int k = 0; k < 784; ++k) {
                    weightsL1[k + j * 784] -= learningRate * D3[j] * input[i][k];
                }
                biasesL1[j] -= learningRate * D3[j];
            }
        }

        std::cout << "Epoch " << e + 1 << " - Average Loss: " << total_loss / nInputs << std::endl;

        if (total_loss / nInputs < tolerance) {
            std::cout << "Early stopping at epoch " << e + 1 << " due to loss tolerance." << std::endl;
            break;
        }

        if (total_loss / nInputs - previous < differenceThreshhold) {
            std::cout << "Early stopping at epoch " << e + 1 << " due to slow improvement in loss." << std::endl;
            break;
        }

		previous = total_loss / nInputs;
    }
}

ForwardPassContainer NetworkMLP::forward_pass(Eigen::Vector<float, 784> image) {

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

	return	{Z1, Z2, Z3};
}

void NetworkMLP::write_model(std::string name) {
	std::ofstream Model("/home/davis/Desktop/Personal/code/DL-Number-Recognition/data/models/" + name);

	Model << "weightsL1\n";
	Model << weightsL1;
	Model << "\nbiasesL1\n";
	Model << biasesL1;
	Model << "\nweightsL2\n";
	Model << weightsL2;
	Model << "\nbiasesL2\n";
	Model << biasesL2;
	Model << "\nweightsL3\n";
	Model << weightsL3;
	Model << "\nbiasesL3\n";
	Model << biasesL3;

	Model.close();
}

int NetworkMLP::predict(Eigen::Vector<float, 784> image) {
	ForwardPassContainer layers = forward_pass(image);
	Eigen::Vector<float, 10> output = layers.Z3;

	int curMax = 0;
	for (int i = 0; i < 10; ++i) {
		if (output[i] > output[curMax]) curMax = i;
	}

	return curMax;

}

float test_accuracy(std::vector<Eigen::Vector<uint8_t, 784>> input, std::vector<int> expectedOutput) {
	/*Tests the accuracy of the model using the test labels*/
	return 1.0;
}
