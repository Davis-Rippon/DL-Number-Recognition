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

	std::cout << "\nInitialising Network..." << std::flush;
	NetworkMLP networkMLP = NetworkMLP();

	/*not working*/
	std::cout << "\nForward Pass..." << std::endl;
	for (int i = 0; i < 10; ++i) {
		show_number(images, i);
		std::cout << networkMLP.forward_pass(images[i + 5000]) << std::endl;
		std::cout << "Break." << std::endl;
	}

	/*std::cout << "\nBack Propagation..." << std::endl;*/
	/*networkMLP.back_prop(images, labels, 0.05, 10);*/

	return 0;
}
