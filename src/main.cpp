#include <vector>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

typedef struct {
	double weight;
	double deltaWeight;
} Connection;

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);

	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal() const { return m_outputVal; }

	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	double m_outputVal;
	double m_weightedOutVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;

	static double randomWeight() { return rand() / double(RAND_MAX); }
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double sumDOW(const Layer &nextLayer) const;

	static double eta;
	static double alpha;
};

double Neuron::eta = 0.1;
double Neuron::alpha = 0.5;

double Neuron::transferFunction(double x)
{
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	return 1.0 - x * x;
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; c++)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
		m_outputWeights.back().deltaWeight = 0.0;
	}

	m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;
	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		sum += prevLayer[n].getOutputVal()*prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_weightedOutVal = sum;
	m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
	// Loss formula: 0.5*(output - target)^2
	double delta = targetVal - m_outputVal;
	// Gradient here is the opposite direction
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal); // Pay attention to the input value
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; n++)
	{
		sum += m_outputWeights[n].weight*nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	for (unsigned n = 0; n < prevLayer.size(); n++)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient
			+ alpha * oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	void printError() const { cout << "error: " << m_error << endl; }

private:
	vector<Layer> m_layers;
	double m_error;
};

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size(); n++)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

Net::Net(const vector<unsigned>& topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++)
	{
		m_layers.push_back(Layer());
		// Last layer has no output
		unsigned numOutputs = layerNum == numLayers - 1 ? 0 : topology[layerNum + 1];

		// "<=" for bias neuron
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
		}

		// Set the bias neuron value to 1.0
		m_layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	// Not include the bias neuron
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Set first layer's neuron's value with input values
	for (unsigned i = 0; i < inputVals.size(); i++)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		// The bias neuron's output is always 1.0, so don't need to calculate
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++)
		{
			// Calling neuron's forward method
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const vector<double> &targetVals)
{
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= (outputLayer.size() - 1);
	m_error = sqrt(m_error);

	// Calculate output layer's gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; n++)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate gradients backward from last hidden layer
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); n++)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; n++)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void generate_train_data()
{
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, 9999);  // generates number in the range 0..9999 
	int dice_roll;
	double randNum;

	ofstream ofs("train.txt", ios::out);
	for (unsigned i = 0; i < 10000; i++)
	{
		dice_roll = distribution(generator);
		randNum = dice_roll / 9999.0;
		int x0, x1, y;
		if (randNum > 0.5) x0 = 1;
		else x0 = 0;

		dice_roll = distribution(generator);
		randNum = dice_roll / 9999.0;
		if (randNum > 0.5) x1 = 1;
		else x1 = 0;

		if (x0 == x1) y = 0;
		else y = 1;

		ofs << x0 << " " << x1 << " " << y << endl;
	}
}

typedef struct
{
	double x0;
	double x1;
	double y;
} SingleInput;

int main()
{
	vector<unsigned> topology({ 2,2,1 });
	Net net(topology);

	ifstream ifs("train.txt", ios::in);
	if (!ifs.is_open()) return -1;

	vector<SingleInput> trainData;
	for (string line; getline(ifs, line);)
	{
		istringstream parseLine(line);
		SingleInput indata;
		parseLine >> indata.x0 >> indata.x1 >> indata.y;
		trainData.push_back(indata);
	}

	vector<double> inputVals, targetVals, resultVals;
	for (unsigned epoch = 0; epoch < 1000; epoch++)
	{
		inputVals = { trainData[epoch].x0, trainData[epoch].x1 };
		targetVals = { trainData[epoch].y };

		net.feedForward(inputVals);
		net.getResults(resultVals);
		net.backProp(targetVals);

		printf("in: %d %d, target: %d, out: %f\n", (int)(inputVals[0]), (int)(inputVals[1]), (int)(targetVals[0]), resultVals[0]);
		net.printError();
	}

	return 0;
}

