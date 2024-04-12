import nengo
from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask
import numpy as np
from nengo_fpga.networks import FpgaPesEnsembleNetwork
from PIL import Image

def resize_img(img, _im_size, _im_size_new):
    img = Image.fromarray(img.reshape((_im_size, _im_size)) * 256, "F")
    img = img.resize((_im_size_new, _im_size_new), Image.LANCZOS)
    return np.array(img.getdata(), np.float32) / 256.0

def one_hot(labels, c=None):
    assert labels.ndim == 1
    n = labels.shape[0]
    c = len(np.unique(labels)) if c is None else c
    y = np.zeros((n, c))
    y[np.arange(n), labels] = 1
    return y

def load_and_process_mnist(reduction_factor=2):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train = 2 * x_train - 1  # Normalize
    x_test = 2 * x_test - 1

    im_size = int(np.sqrt(x_train.shape[1]))  # Image size
    if reduction_factor > 1:
        im_size_new = int(im_size // reduction_factor)
        x_train_resized = np.zeros((x_train.shape[0], im_size_new**2))
        x_test_resized = np.zeros((x_test.shape[0], im_size_new**2))
        for i in range(x_train.shape[0]):
            x_train_resized[i, :] = resize_img(x_train[i], im_size, im_size_new)
        for i in range(x_test.shape[0]):
            x_test_resized[i, :] = resize_img(x_test[i], im_size, im_size_new)
        x_train, x_test, im_size = x_train_resized, x_test_resized, im_size_new
    train_targets = one_hot(y_train, 10)
    test_targets = one_hot(y_test, 10)
    return x_train, y_train, x_test, y_test, train_targets, test_targets, im_size


def create_model(x_train, x_test, im_size, n_out, n_vis, n_hid, presentation_time, train_targets, rng,
                 neuron_type=nengo.neurons.RectifiedLinear(), synapse_time=None):
    """
    Create a Nengo model for MNIST classification with customizable neuron model and synapse time.

    Parameters:
    x_train (array): Training data.
    x_test (array): Testing data.
    im_size (int): Image size (dimension of one side).
    n_out (int): Number of output classes.
    n_vis (int): Number of visible units (input dimensionality).
    n_hid (int): Number of hidden neurons.
    presentation_time (float): Duration for which each input is presented.
    train_targets (array): Target output for training.
    rng (np.random.RandomState): Random state for reproducibility.
    neuron_type (nengo.Neurons, optional): Neuron model to use.
    synapse_time (float, optional): Synaptic time constant for connections.

    Returns:
    tuple: Tuple containing the model and input, output, and probe nodes.
    """
    gabor_size = (int(im_size / 2.5), int(im_size / 2.5))  # Size of the Gabor filters

    # Generate the encoders
    encoders = Gabor().generate(n_hid, gabor_size, rng=rng)
    encoders = Mask((im_size, im_size)).populate(encoders, rng=rng, flatten=True)

    # Ensemble parameters
    max_firing_rates = 200
    ens_intercepts = nengo.dists.Choice([-0.5])
    ens_max_rates = nengo.dists.Choice([max_firing_rates])

    # Output connection parameters
    conn_eval_points = x_train
    conn_function = train_targets
    conn_solver = nengo.solvers.LstsqL2(reg=0.01)

    model = nengo.Network(label="MNIST Classification Network", seed=3)
    with model:
        # Visual input (the MNIST images) to the network
        input_node = nengo.Node(
            nengo.processes.PresentInput(x_test, presentation_time), label="Input MNIST Images"
        )

        ens = FpgaPesEnsembleNetwork(
            None,
            n_neurons=n_hid,
            dimensions=n_vis,
            learning_rate=0,
            function=conn_function,
            eval_points=conn_eval_points,
            label="Output Class"
        )

        # Set custom ensemble parameters
        ens.ensemble.neuron_type = neuron_type
        ens.ensemble.intercepts = ens_intercepts
        ens.ensemble.max_rates = ens_max_rates
        ens.ensemble.encoders = encoders

        # Set custom connection parameters
        ens.connection.synapse = synapse_time
        ens.connection.solver = conn_solver

        # Output display node
        output_node = nengo.Node(size_in=n_out, label="Output Class")
        output_probe = nengo.Probe(output_node, synapse=synapse_time)

        # Connections from input node to FPGA ensemble, and from FPGA ensemble to output node
        nengo.Connection(input_node, ens.input, synapse=synapse_time)
        nengo.Connection(ens.output, output_node, synapse=synapse_time)

    return model, input_node, output_node, output_probe
