# NeuralNetwork
<br>
In no way am I actually qualified. Sad!
<br>
<br>
This takes advantage of mxjava, a Matrix Multiplication function library that I have created. Yay?
<br>
<br>
Future projects based on this will come out in the near future, stay tuned. <br> <br>
<br> <b> #1 Boring Project: Language Predictor! </b>
<br> This has gone through a lot of updates - the very new update can reach crazy accurate values. Notice the difference in input from the first experiment (LanguageDetector.java) and the recently updated one (LanguageDetector2.java).
<br> <br> 
<h1> IMPLEMENTATION OF THE NEURAL NETWORK </h1>
<br><br>
The Neural Network with only one hidden layer (named TwoLayerNeuralNetwork.java) is designed so that there is a training rate [not implemented for the two hidden layer neural net]. This class is implemented as such: <br> <br>
TwoLayerNeuralNetwork a = new TwoLayerNeuralNetwork(INPUT,OUTPUT,hiddenNeuronsPerLayer,learningRate);
<br> INPUT = an array of arrays. Each instance is an array with multiple inputs.
<br> OUTPUT = an array of arrays. Each instance is an array of the output.
<br> hiddenNeuronsPerLayer = Hidden Layers! 
<br> learningRate = Learning Rate is a number from 0-1.
