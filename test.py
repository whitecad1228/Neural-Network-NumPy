import math

import numpy as np

import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib

font = {'weight': 'normal',
        'size': 22}
matplotlib.rc('font', **font)

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
np.random.seed(90)
step_size = .001
dec_step_size = 0.0001
batch_size = 100
max_epochs = 2000

# GLOBAL PARAMETERS FOR NETWORK ARCHITECTURE
number_of_layers = 6
width_of_layers = 30  # only matters if number of layers > 1
# activation = "ReLU" if False else "Sigmoid"
activation = "ReLU"
weightDecay = False


def main():
    # Load data and display an example
    X_train, Y_train, X_val, Y_val, X_test = loadData()
    displayExample(X_train[np.random.randint(0, len(X_train))])

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below
    net = FeedForwardNeuralNetwork(X_train.shape[1], 10, width_of_layers, number_of_layers, activation=activation, keep_rate=0.6)

    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    # Loss function
    lossFunc = CrossEntropySoftmax()

    # Indicies we will use to shuffle data randomly
    inds = np.arange(len(X_train))
    for i in range(max_epochs):

        # Shuffled indicies so we go through data in new random batches
        np.random.shuffle(inds)

        # Go through all datapoints once (aka an epoch)
        j = 0
        acc_running = loss_running = 0
        while j < len(X_train):
            # Select the members of this random batch
            b = min(batch_size, len(X_train) - j)
            X_batch = X_train[inds[j:j + b]]
            Y_batch = Y_train[inds[j:j + b]].astype(int)

            # Compute the scores for our 10 classes using our model
            logits = net.forward(X_batch)
            loss = lossFunc.forward(logits, Y_batch)
            acc = np.mean(np.argmax(logits, axis=1)[:, np.newaxis] == Y_batch)

            # Compute gradient of Cross-Entropy Loss with respect to logits
            loss_grad = lossFunc.backward()

            # Pass gradient back through networks
            net.backward(loss_grad)

            # Take a step of gradient descent
            if weightDecay:
                aug_step_size = step_size / 10 * (math.floor(i / 1000) + 1)
                net.step(aug_step_size)
            else:
                net.step(step_size)

            # Record losses and accuracy then move to next batch
            losses.append(loss)
            accs.append(acc)
            loss_running += loss * b
            acc_running += acc * b

            j += batch_size

        # Evaluate performance on validation. This function looks very similar to the training loop above,
        vloss, vacc = evaluateValidation(net, X_val, Y_val, batch_size)
        val_losses.append(vloss)
        val_accs.append(vacc)

        # Print out the average stats over this epoch
        lossInfo = loss_running / len(X_train)
        trainAccInfo = acc_running / len(X_train) * 100
        logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(i,lossInfo,trainAccInfo,vacc * 100))

    fig, ax1 = plt.subplots(figsize=(16, 9))
    color = 'tab:red'
    ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
    ax1.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_losses))], val_losses, c="red",
             label="Val. Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.01, 3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_accs))], val_accs, c="blue",
             label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 1.01)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.show()

    # ################################
    # # Q7 Evaluate on Test
    # ################################
    inds = np.arange(len(X_test), dtype=int)
    # print("inds.shape",inds.shape)
    Y_test = np.empty(len(X_test))
    # print("Y_test.shape",Y_test.shape)
    # Go through all datapoints once (aka an epoch)
    j = 0
    acc_running = loss_running = 0
    while j < len(X_test):
        # Select the members of this random batch
        b = min(batch_size, len(X_test) - j)
        X_batch = X_test[inds[j:j + b]]
        # Y_batch = Y_train[inds[j:j + b]].astype(int)
        # print("X_test",X_test)
        # print(Y_batch)

        # Compute the scores for our 10 classes using our model
        logits = net.forward(X_batch)
        Y_test[inds[j:j + b]] = np.argmax(logits, axis=1)

        test = np.argmax(logits, axis=1)
        # print("logits.shape",logits.shape)
        # print("logits[0]",logits[0])
        # print("test[0]",test[0])
        # print("test.shape",test.shape)
        # print("Y-test",Y_test.shape)

        j += batch_size

    # print(Y_test.shape)
    # print(inds.shape)
    # print(np.expand_dims(test_out,axis=0))
    # print(Y_test[0])
    print(inds[0])
    test_out = np.concatenate((np.expand_dims(inds, axis=1), np.expand_dims(Y_test, axis=1)), axis=1)
    # print(test_out.shape)
    print("testout[0]", test_out[0])
    test_out = test_out.astype(int)
    print("testout[0]", test_out[0])
    header = np.array([["id", "digit"]])
    test_out = np.concatenate((header, test_out))
    print("testout[0]", test_out[1])
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')


class LinearLayer:

    def __str__(self):
        return "LL"

    # Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
    def __init__(self, input_dim, output_dim, keep_rate):
        # print("input_dim:", input_dim,"output_dim:",output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias = np.ones((1, output_dim)) * 0.5
        self.keep_rate = keep_rate
        self.mask = np.zeros((output_dim,1))

    # During the forward pass, we simply compute Xw+b
    def forward(self, input):

        self.input = input  # Storing X
        self.mask = np.random.rand(self.output_dim) < self.keep_rate
        # print("input dim", self.input.shape)
        # print("weights dim", self.weights.shape)
        # print("mask",self.mask)
        # print("weights * mask ", self.input @ self.weights)
        # print("test:", (self.input @ (self.weights * self.mask)) + (self.bias*self.mask))
        output = (self.input @ (self.weights * self.mask)) + (self.bias*self.mask)/self.keep_rate
        # output = (((self.input @ self.weights) + self.bias)*self.mask)/self.keep_rate
        #output = (self.input @ ((self.weights + self.bias) * self.mask))/self.keep_rate
        # print("output",output)
        # print("output shape",output.shape)
        # print("OGoutput",OGoutput)
        return output

    def eval(self,input):
        self.input = input  # Storing X
        # print(self.weights)
        return self.input @ self.weights + self.bias
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Q3 Implementing Backward Pass for Linear
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Inputs :
    #
    # grad dL / dZ -- For a batch size of n , grad is a ( n x output_dim ) matrix where
    # the i ’ th row is the gradient of the loss of example i with respect
    # to z_i ( the output of this layer for example i )
    #
    # Computes and stores :
    #
    # self . grad_weights dL / dW -- A ( input_dim x output_dim ) matrix storing the
    # gradient of the loss with respect to the weights of this layer .
    #
    # self . grad_bias dL / db - - A (1 x output_dim ) matrix storing the gradient
    # of the loss with respect to the bias of this layer .
    #
    # Return Value :
    #
    # grad_input dL / dX -- For a batch size of n , grad_input is a ( n x input_dim )
    # matrix where the i ’ th row is the gradient of the loss of
    # example i with respect to x_i ( the input of this layer for example i )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def backward(self, grad):
        # print("mask",self.mask)
        self.grad_weights = ((self.input.T @ grad) * self.mask)
        # print("self.grad_weights", self.grad_weights)
        self.grad_bias = (np.sum(grad, axis=0) * self.mask)
        # print("self.grad_bias",self.grad_bias)
        # print("out 1:",(grad @ (self.weights * self.mask).T)/self.keep_rate)
        # print("out 2:",(grad @ self.weights.T)/self.keep_rate)
        # print("out 3",self.weights.T)
        # print("out 4:",(self.weights * self.mask).T )
        output = (grad @ (self.weights * self.mask).T)/self.keep_rate
        # print("output",output)
        # print("weights",self.weights)
        # print("grad",grad)
        # output = (grad @ self.weights.T)/self.keep_rate
        return output

    def step(self, step_size):
        self.weights -= step_size * self.grad_weights
        self.bias -= step_size * self.grad_bias


###############################################
# Instructor code below. Worth understanding
# but don't need to modify for base assignment
###############################################


class FeedForwardNeuralNetwork:

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation="ReLU",keep_rate = 1):

        if num_layers == 1:
            self.layers = [LinearLayer(input_dim, output_dim,1)]
        else:
            self.layers = [LinearLayer(input_dim, hidden_dim,1)]
            self.layers.append(Sigmoid() if activation == "Sigmoid" else ReLU())
            for i in range(num_layers - 2):
                self.layers.append(LinearLayer(hidden_dim, hidden_dim,keep_rate))
                self.layers.append(Sigmoid() if activation == "Sigmoid" else ReLU())
            self.layers.append(LinearLayer(hidden_dim, output_dim,1))

        # print(self.layers)

    def forward(self, X):
        # print("******************************* Forward ************************************")
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        # print("******************************* BACK ************************************")
        for layer in reversed(self.layers):
            # print("layer:", layer)
            grad = layer.backward(grad)

    def step(self, step_size=0.001):
        # print("******************************* STEP ************************************")
        for layer in self.layers:
            layer.step(step_size)

    def eval(self, X):
        for layer in self.layers:
            X = layer.eval(X)
        return X

# Sigmoid or Logistic Activation Function
class Sigmoid:

    def __str__(self):
        return "Sigmoid"

    # Given the input, apply the sigmoid function
    # store the output value for use in the backwards pass
    def forward(self, input):
        self.act = 1 / (1 + np.exp(-input))
        return self.act

    def eval(self, input):
        self.act = 1 / (1 + np.exp(-input))
        return self.act

    # Compute the gradient of the output with respect to the input
    # self.act*(1-self.act) and then multiply by the loss gradient with
    # respect to the output to produce the loss gradient with respect to the input
    def backward(self, grad):
        return grad * self.act * (1 - self.act)

    # The Sigmoid has no parameters so nothing to do during a gradient descent step
    def step(self, step_size):
        return


# Rectified Linear Unit Activation Function
class ReLU:

    def __str__(self):
        return "ReLU"
    # Forward pass is max(0,input)
    def forward(self, input):
        self.mask = (input > 0)
        return input * self.mask

    def eval(self, input):
        self.mask = (input > 0)
        return input * self.mask

    # Backward pass masks out same elements
    def backward(self, grad):
        return grad * self.mask

    # No parameters so nothing to do during a gradient descent step
    def step(self, step_size):
        return


#####################################################
# Utility Functions for Computing Loss / Val Metrics
#####################################################
def softmax(x):
    x -= np.max(x, axis=1)[:, np.newaxis]  # Numerical stability trick
    return np.exp(x) / (np.sum(np.exp(x), axis=1)[:, np.newaxis])


class CrossEntropySoftmax:

    def forward(self, logits, labels):
        self.probs = softmax(logits)
        self.labels = labels
        return -np.mean(np.log(self.probs[np.arange(len(self.probs))[:, np.newaxis], labels] + 0.00001))

    def backward(self):
        grad = self.probs
        grad[np.arange(len(self.probs))[:, np.newaxis], self.labels] -= 1
        return grad.astype(np.float64) / len(self.probs)


def evaluateValidation(model, X_val, Y_val, batch_size,):
    val_loss_running = 0
    val_acc_running = 0
    j = 0

    lossFunc = CrossEntropySoftmax()

    while j < len(X_val):
        b = min(batch_size, len(X_val) - j)
        X_batch = X_val[j:j + b]
        Y_batch = Y_val[j:j + b].astype(int)

        logits = model.eval(X_batch)
        loss = lossFunc.forward(logits, Y_batch)
        acc = np.mean(np.argmax(logits, axis=1)[:, np.newaxis] == Y_batch)

        val_loss_running += loss * b
        val_acc_running += acc * b

        j += batch_size

    return val_loss_running / len(X_val), val_acc_running / len(X_val)


#####################################################
# Utility Functions for Loading and Displaying Data
#####################################################
def loadData(normalize=True):
    train = np.loadtxt("mnist_small_train.csv", delimiter=",", dtype=np.float64)
    val = np.loadtxt("mnist_small_val.csv", delimiter=",", dtype=np.float64)
    test = np.loadtxt("mnist_small_test.csv", delimiter=",", dtype=np.float64)

    # Normalize Our Data
    if normalize:
        X_train = train[:, :-1] / 256 - 0.5
        X_val = val[:, :-1] / 256 - 0.5
        X_test = test / 256 - 0.5
    else:
        X_train = train[:, :-1]
        X_val = val[:, :-1]
        X_test = test

    Y_train = train[:, -1].astype(int)[:, np.newaxis]
    Y_val = val[:, -1].astype(int)[:, np.newaxis]

    logging.info("Loaded train: " + str(X_train.shape))
    logging.info("Loaded val: " + str(X_val.shape))
    logging.info("Loaded test: " + str(X_test.shape))

    return X_train, Y_train, X_val, Y_val, X_test


def displayExample(x):
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()