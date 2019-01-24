from utils import *


X, y = generate_dataset(n_classes=n_classes)
# plot_dataset(X, y)
model = build_model(X, y, n_hidden_dim)
plot_decision_boundary(model, X, y)
print(X)
