from utils import *
from variables import *
import os


X, y = generate_dataset(n_classes=n_classes, n_examples=n_examples)
# plot_dataset(X, y, n_classes)
for hdim1 in n_hidden_dim:
    for hdim2 in n_hidden_dim:
        nn_structure = (n_input_dim, hdim1, hdim2, n_classes)
        for mb_size in minibatch_size:
            nn = NeuralNetwork(activation_function, X, y, n_examples, n_input_dim, n_classes)
            model = nn.build_model(nn_hdim=[hdim1, hdim2], num_passes=n_passes, lr=epsilon, \
                                    print_loss=print_loss_value, minibatch_size=mb_size, \
                                    reduce_lr=lr_annealing, decay=decay_rate, reg_lambda=reg_lambda)
            plot_dir = PLOT + "{} activation/{} batch size/{} annealing lr/".format(activation_function[0], mb_size, lr_annealing)
            plot_name = "structure={},minibatch_size={},annealing_lr={},activation={}".format(nn_structure, mb_size, lr_annealing, activation_function)
            plot_name += ".png"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            nn.plot_decision_boundary(model, plot_dir + plot_name)
# print(X)
