n_classes = 2
n_examples = 200
n_input_dim = 2
n_output_dim = 2
n_hidden_dim = 40
n_hidden_layers = 1
n_hidden_dim2 = 0
n_passes = 20000
print_loss_value = True
minibatch_size = n_examples

epsilon = 0.01
reg_lambda = 0.01
lr_annealing = True
decay = epsilon / n_passes

activation_function1 = "tanh"
activation_function2 = "softmax"

PLOT = "./plots/"
if n_hidden_layers == 1:
    plot_name = "minibatch_size={},structure={}-{}-{},annealing_lr.png".format(minibatch_size, n_input_dim, n_hidden_dim, n_output_dim)
else:
    plot_name = "minibatch_size={},structure={}-{}-{}-{},annealing_lr.png".format(minibatch_size, n_input_dim, n_hidden_dim, n_hidden_dim2, n_output_dim)
