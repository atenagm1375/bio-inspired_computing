from utils import *
from variables import *


X, y = generate_dataset(n_classes=n_classes, n_examples=n_examples)
# plot_dataset(X, y)
for hdim in n_hidden_dim:
    nn_structure = (n_classes, hdim, n_classes)
    for mb_size in minibatch_size:
        nn = NeuralNetwork(activation_function, X, y, n_examples, n_classes)
        model = nn.build_model(nn_hdim=hdim, num_passes=n_passes, lr=epsilon, \
                                print_loss=print_loss_value, minibatch_size=mb_size, \
                                reduce_lr=lr_annealing, decay=decay_rate, reg_lambda=reg_lambda)
        plot_name = "structure={},minibatch_size={},annealing_lr={},activation={}".format(nn_structure, mb_size, lr_annealing, activation_function)
        plot_name += ".png"
        nn.plot_decision_boundary(model, PLOT + plot_name)
# print(X)
