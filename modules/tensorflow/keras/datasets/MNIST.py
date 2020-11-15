from tensorflow.keras.datasets import mnist, fashion_mnist


def import_mnist(variant='mnist', shape='channels', binarization=False):
    """Load either MNIST or Fashion-MNIST with a specified shape and normalization."""
    # (images, labels)
    dataset = mnist
    if variant == 'fashion':
        dataset = fashion_mnist
    (x, x_labels), (y, y_labels) = dataset.load_data()

    # normalization: (0,1), float32
    x = x.astype('float32') / 255.
    y = y.astype('float32') / 255.

    if binarization:
        x[x >= .5] = 1.
        x[x < .5] = 0.
        y[y >= .5] = 1.
        y[y < .5] = 0.

    # shape
    D = (-1, 28, 28, 1)
    if shape == '1D':
        D = (-1, 784)
    elif shape == '2D':
        D = (-1, 28, 28)
    x = x.reshape(D)
    y = y.reshape(D)

    return (x, x_labels), (y, y_labels)
