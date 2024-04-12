import logging
import nengo
import numpy as np
from nengo_extras.gui import image_display_function
from model_def import create_model, load_and_process_mnist

# Adjust your parameter here
presentation_time = 0.05
neuron_model = nengo.neurons.LIF()
synapse_time = 0.005

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

x_train, y_train, x_test, y_test, train_targets, test_targets, im_size = load_and_process_mnist()

# Set up the vision network parameters
n_vis = x_train.shape[1]  # Number of training samples
n_out = train_targets.shape[1]  # Number of output classes
n_hid = 16000 // (im_size**2)  # Number of neurons to use

n_test = x_test.shape[0]
simulation_time = presentation_time * n_test

rng = np.random.RandomState(9)

predicted_labels = []
actual_labels = []
model, input_node, output_node, output_probe = create_model(x_train, x_test, im_size, n_out, n_vis, n_hid, presentation_time, train_targets, rng, neuron_model, synapse_time)



def max_index(t, x):

    return [np.argmax(x)]


def combine_index_and_label(t, x):
    current_index = int(t / presentation_time) % len(y_test)
    actual_label = y_test[current_index]
    return np.hstack([x, [actual_label]])


def show_digit(t, x):
    if not hasattr(show_digit, "last_time_printed"):
        show_digit.last_time_printed = -1

    time_since_last_print = t - show_digit.last_time_printed
    if time_since_last_print >= 1.5*presentation_time or show_digit.last_time_printed == -1:
        predicted_index = int(x[0])  # The predicted index is the first element in the array
        actual_label = int(x[1])  # The actual label is the second element
        digits = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
        print(f"Time {t:.2f}: Predicted Digit - {digits[predicted_index]}, Actual Digit - {digits[actual_label]}")
        show_digit.last_time_printed = t + presentation_time//2

        predicted_labels.append(predicted_index)
        actual_labels.append(actual_label)



with model:
    # Input image display (for nengo_gui)
    image_shape = (1, im_size, im_size)
    display_func = image_display_function(image_shape, offset=1, scale=128)
    display_node = nengo.Node(display_func, size_in=input_node.size_out)
    nengo.Connection(input_node, display_node, synapse=None)

    # Output SPA display (for nengo_gui)
    vocab_names = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
    vocab_vectors = np.eye(len(vocab_names))
    vocab = nengo.spa.Vocabulary(len(vocab_names))
    for name, vector in zip(vocab_names, vocab_vectors):
        vocab.add(name, vector)

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        output_spa = nengo.spa.State(len(vocab_names), subdimensions=n_out, vocab=vocab)
    nengo.Connection(output_node, output_spa.input)

    max_index_node = nengo.Node(max_index, size_in=n_out, size_out=1)
    combined_node = nengo.Node(combine_index_and_label, size_in=1, size_out=2)
    display_digit_node = nengo.Node(show_digit, size_in=2, size_out=0)

    nengo.Connection(output_node, max_index_node, synapse=None)
    nengo.Connection(max_index_node, combined_node, synapse=None)
    nengo.Connection(combined_node, display_digit_node, synapse=None)
