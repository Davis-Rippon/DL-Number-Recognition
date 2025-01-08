#include "functions.h"
#include "NetworkMLP.h"
#include <iostream>
#include <iomanip>

int main() {
	std::string imageDatabase = "data/train-images-idx3-ubyte";
	std::string labelDatabase = "data/train-labels-idx1-ubyte";

	std::cout << "Reading Labels...\n" << std::flush;
	std::vector<int> labels = read_labels(labelDatabase, 60000);
	std::cout << "\nDone " << std::endl;

	std::cout << "Reading Images...\n" << std::flush;
	std::vector<Eigen::Vector<float, 784>> images = read_images(imageDatabase, 60000);
	std::cout << "\nDone " << std::endl;

	std::string testImageDatabase = "data/t10k-images-idx3-ubyte";
	std::string testLabelDatabase = "data/t10k-labels-idx1-ubyte";

	std::cout << "Reading Test Labels...\n" << std::flush;
	std::vector<int> testLabels = read_labels(testLabelDatabase, 10000);
	std::cout << "\nDone " << std::endl;

	std::cout << "Reading Test Images...\n" << std::flush;
	std::vector<Eigen::Vector<float, 784>> testImages = read_images(testImageDatabase, 10000);
	std::cout << "\nDone " << std::endl;

	std::cout << "\nInitialising Network..." << std::flush;
	NetworkMLP networkMLP = NetworkMLP();

	/*not working*/
	std::cout << "\nForward Pass..." << std::endl;

	std::cout << "\nBack Propagation..." << std::endl;
	networkMLP.back_prop(images, labels, 0.05, 0.01, 0.01, 25);

	for (int i = 0; i < 10; ++i) {
		show_number(testImages, i);
		std::cout << "Model Prediction: " << networkMLP.predict(testImages[i]) << std::endl;
	}

	return 0;
}
