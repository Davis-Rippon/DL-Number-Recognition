#include <eigen3/Eigen/Dense>

struct ForwardPassContainer {
	Eigen::Vector<float, 16> Z1;
	Eigen::Vector<float, 16> Z2;
	Eigen::Vector<float, 10> Z3;
};

class NetworkMLP {
	public:
		NetworkMLP();
		ForwardPassContainer forward_pass(Eigen::Vector<float, 784> input);
		void back_prop(std::vector<Eigen::Vector<float, 784>> input, std::vector<int> expectedOutput, float learningRate, float tolerance, float differenceThreshhold, int epochs=50);
		uint8_t test_accuracy(std::vector<Eigen::Vector<float, 784>> input, std::vector<int> expectedOutput);
		int predict(Eigen::Vector<float, 784> image);
	private:
		float loss_function(Eigen::Vector<float, 10> output, int expected);
		Eigen::Vector<float, 12544> weightsL1;
		Eigen::Vector<float, 16> biasesL1;
		
		Eigen::Vector<float, 256> weightsL2;
		Eigen::Vector<float, 16> biasesL2;
		
		Eigen::Vector<float, 160> weightsL3;
		Eigen::Vector<float, 10> biasesL3;
};
