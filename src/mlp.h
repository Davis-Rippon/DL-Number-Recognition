#include <eigen3/Eigen/Dense>

class MLP {
	public:
		int forward_pass(Eigen::MatrixXd input);
		void back_prop(Eigen::MatrixXd input, Eigen::MatrixXd expectedOutput);
	
	private:
		/* Each layer has a set of weights and biases */
		Eigen::MatrixXd W1;
		Eigen::MatrixXd b1;
		Eigen::MatrixXd W2;
		Eigen::MatrixXd b2;
		Eigen::MatrixXd W3;
		Eigen::MatrixXd b3;
		
};
