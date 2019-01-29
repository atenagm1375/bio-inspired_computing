n_classes = 2
n_examples = 200
n_input_dim = n_classes
n_output_dim = n_classes
n_hidden_dim = [1, 2, 3, 4, 5, 20, 40]
n_hidden_layers = 1
n_passes = 20000
print_loss_value = True
minibatch_size = [n_examples, n_examples // 2, n_examples // 4, n_examples // 5, n_examples // 10]

epsilon = 0.01
reg_lambda = 0.01
lr_annealing = True
decay_rate = epsilon / n_passes

activation_function = ["leaky ReLU", "softmax"]

PLOT = "./plots/"
