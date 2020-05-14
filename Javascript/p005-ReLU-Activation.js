const math = require("mathjs");

const X = math.matrix([
	[1.0, 2.0, 3.0, 2.5],
	[2.0, 5.0, -1.0, 2.0],
	[-1.5, 2.7, 3.3, -0.8]]);

class Layer_Dense {
	constructor (n_inputs, n_neurons) {
		this.weights = math.random([n_inputs, n_neurons], -1.0, 1.0);
		this.biases = math.zeros(1, n_neurons);
	}
	
	forward (inputs) {
		var biasesmat = this.biases;
		// Since only adding matrices elementwise is supported, you need to make the biases into a matrix and not a vector.
		for (var i=0; i<inputs.size()[0]-1;i++) {biasesmat=math.concat(biasesmat, this.biases, 0);}
		this.output = math.add(math.multiply(inputs, this.weights), biasesmat);
	}
}

class Activation_ReLU {
	forward(inputs) {
		this.output = math.map(inputs,x => math.max(0,x));
	}
}

var layer1 = new Layer_Dense(4, 5);
var layer2 = new Layer_Dense(5, 2);

var activation1 = new Activation_ReLU();
var activation2 = new Activation_ReLU();

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

console.table(activation2.output._data)
