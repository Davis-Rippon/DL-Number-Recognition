#include <eigen3/Eigen/Dense>

class NetworkMLP {
	public:
		NetworkMLP();
		int forward_pass(Eigen::Vector<uint8_t, 784> image);
		void back_prop(std::vector<Eigen::Vector<uint8_t, 784>> input, std::vector<int> expectedOutput, float learningRate, int tolerance);
		uint8_t test_accuracy(std::vector<Eigen::Vector<uint8_t, 784>> input, std::vector<int> expectedOutput);
	private:
		Eigen::Vector<uint8_t, 12544> weightsL1;
		Eigen::Vector<uint8_t, 16> biasesL1;
		
		Eigen::Vector<uint8_t, 256> weightsL2;
		Eigen::Vector<uint8_t, 16> biasesL2;
		
		Eigen::Vector<uint8_t, 160> weightsL3;
		Eigen::Vector<uint8_t, 10> biasesL3;
};
