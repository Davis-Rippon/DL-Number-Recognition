#include <eigen3/Eigen/Dense>

class NetworkMLP {
	public:
		NetworkMLP();
		Eigen::Vector<float, 10> forward_pass(Eigen::Vector<float, 784> input);
		void back_prop(std::vector<Eigen::Vector<float, 784>> input, std::vector<int> expectedOutput, float learningRate, int tolerance, int epochs=3);
		uint8_t test_accuracy(std::vector<Eigen::Vector<float, 784>> input, std::vector<int> expectedOutput);
	private:
		float loss_function(Eigen::Vector<float, 10> output, int expected);
		Eigen::Vector<float, 12544> weightsL1;
		Eigen::Vector<float, 16> biasesL1;
		
		Eigen::Vector<float, 256> weightsL2;
		Eigen::Vector<float, 16> biasesL2;
		
		Eigen::Vector<float, 160> weightsL3;
		Eigen::Vector<float, 10> biasesL3;
};
