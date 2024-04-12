import logging
import nengo
import numpy as np
from model_def import create_model, load_and_process_mnist
from sklearn.metrics import accuracy_score


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

x_train, y_train, x_test, y_test, train_targets, test_targets, im_size = load_and_process_mnist()

# Set up the vision network parameters
n_vis = x_train.shape[1]  # Number of training samples
n_out = train_targets.shape[1]  # Number of output classes
n_hid = 16000 // (im_size**2)  # Number of neurons to use
presentation_time = 0.05

n_test = x_test.shape[0]
simulation_time = presentation_time * n_test

rng = np.random.RandomState(9)
# model, input_node, output_node, output_probe = create_model(x_train, x_test, im_size, n_out, n_vis, n_hid, presentation_time, train_targets, rng)
model, input_node, output_node, output_probe = create_model(x_train, x_test, im_size, n_out, n_vis, n_hid, presentation_time, train_targets, rng, nengo.neurons.LIF(), 0.005)


def max_index(t, x):
    return [np.argmax(x)]


def combine_index_and_label(t, x):
    current_index = int(t / presentation_time) % len(y_test)
    actual_label = y_test[current_index]
    return np.hstack([x, [actual_label]])

predicted_labels = []
actual_labels = []

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
    max_index_node = nengo.Node(max_index, size_in=n_out, size_out=1)
    combined_node = nengo.Node(combine_index_and_label, size_in=1, size_out=2)
    display_digit_node = nengo.Node(show_digit, size_in=2, size_out=0)

    nengo.Connection(output_node, max_index_node, synapse=None)
    nengo.Connection(max_index_node, combined_node, synapse=None)
    nengo.Connection(combined_node, display_digit_node, synapse=None)

    with nengo.Simulator(model) as sim:
        sim.run(simulation_time)  # Run the simulation for 6 minutes

# Calculate accuracy using only the first 10000 data points
accuracy = accuracy_score(actual_labels[:10000], predicted_labels[:10000])
print("Accuracy of the model:", accuracy)