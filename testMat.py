import numpy as np

if __name__ == '__main__':
    input_dim = 6
    output_dim = 5
    weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
    bias = np.random.rand(1, output_dim)
    print(bias)
    print(weights)
    T1 = weights + bias
    print("weights + bias",T1)
    mask = np.random.rand(output_dim) < 0.8
    print(mask)
    print(mask.shape)
    print(T1.shape)
    print(T1 * mask)

