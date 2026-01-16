import torch

# Dataset
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y_and = torch.tensor([[0.],[0.],[0.],[1.]])
y_or  = torch.tensor([[0.],[1.],[1.],[1.]])

# Perceptron training with given learning rate
def train_with_lr(y, lr):
    w = torch.zeros(2,1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    for _ in range(1000):
        y_pred = torch.sigmoid(X @ w + b)
        loss = ((y_pred - y)**2).mean() #mse
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()

    return w, b, loss.item() #it is pytorch tensor 

# Greedy search for best learning rate
def greedy_lr_search(y):
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    best_lr = None
    best_loss = float("inf")
    best_w, best_b = None, None

    for lr in learning_rates:
        w, b, loss = train_with_lr(y, lr)
        print(f"LR = {lr:>5} | Loss = {loss:.6f}")

        if loss < best_loss:
            best_loss = loss
            best_lr = lr
            best_w, best_b = w, b

    print("\nBest Learning Rate =", best_lr)
    return best_w, best_b, best_lr

print("----- AND Gate Greedy Search -----")
w_and, b_and, lr_and = greedy_lr_search(y_and)

print("\nAND Predictions:")
print((torch.sigmoid(X @ w_and + b_and) > 0.5).float())

# OR gate
print("\n----- OR Gate Greedy Search -----")
w_or, b_or, lr_or = greedy_lr_search(y_or)

print("\nOR Predictions:")
print((torch.sigmoid(X @ w_or + b_or) > 0.5).float())
