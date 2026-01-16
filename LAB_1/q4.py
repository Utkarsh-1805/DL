import torch
import pandas as pd

data = pd.read_csv(r"d:/LAB_SEM_6/DL/LAB_1/house_price_full+(2) - house_price_full+(2).csv", header=None)

data.columns = ["bedrooms", "sqft", "price"]
data = data.apply(pd.to_numeric, errors='coerce').dropna()

std_X = data[["bedrooms", "sqft"]].std()
std_y = data["price"].std()

if (std_X == 0).any() or std_y == 0:
    print("Error: Zero standard deviation in data. Cannot normalize.")
    exit()
X = torch.tensor(data[["bedrooms", "sqft"]].values, dtype=torch.float32)
y = torch.tensor(data["price"].values, dtype=torch.float32).view(-1, 1)
X = (X - X.mean(dim=0)) / X.std(dim=0)
y = (y - y.mean()) / y.std()

w11 = torch.randn(1, requires_grad=True)
w21 = torch.randn(1, requires_grad=True)
b1 = torch.zeros(1, requires_grad=True)
w12 = torch.randn(1, requires_grad=True)
w22 = torch.randn(1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
w1 = torch.randn(1, requires_grad=True)
w2 = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.01

for epoch in range(5000):
    x1 = X[:, 0]
    x2 = X[:, 1]
    h1 = torch.relu(w11 * x1 + w21 * x2 + b1)
    h2 = torch.relu(w12 * x1 + w22 * x2 + b2)
    y_pred = w1 * h1 + w2 * h2 + b
    loss = ((y_pred - y.squeeze()) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w11 -= lr * w11.grad
        w21 -= lr * w21.grad
        b1 -= lr * b1.grad
        w12 -= lr * w12.grad
        w22 -= lr * w22.grad
        b2 -= lr * b2.grad
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        b -= lr * b.grad
        w11.grad.zero_()
        w21.grad.zero_()
        b1.grad.zero_()
        w12.grad.zero_()
        w22.grad.zero_()
        b2.grad.zero_()
        w1.grad.zero_()
        w2.grad.zero_()
        b.grad.zero_()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.5f}")

test_house = torch.tensor([[3., 2000.]])
test_house = (test_house - data[["bedrooms", "sqft"]].mean().values) / data[["bedrooms", "sqft"]].std().values
x1, x2 = test_house[0]
h1 = torch.relu(w11 * x1 + w21 * x2 + b1)
h2 = torch.relu(w12 * x1 + w22 * x2 + b2)
price_norm = w1 * h1 + w2 * h2 + b
price = price_norm * data["price"].std() + data["price"].mean()
print("\nPredicted Price for 3 BHK, 2000 sqft:", price.item())