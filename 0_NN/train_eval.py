# train_eval.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from data_prep import prepare_data
import matplotlib.pyplot as plt

def train_model(X_train, y_train, learning_rate=0.01, epochs=100):
    torch.manual_seed(41)
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for i in range(epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        if i % 10 == 0:
            print(f'Epoch {i} loss: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    plt.plot(range(epochs), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    
    return model

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        y_eval = model(X_test)
        loss = nn.CrossEntropyLoss()(y_eval, y_test)
    print(f'Test loss: {loss.item()}')
    
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_val = model(data)
            if y_val.argmax().item() == y_test[i]:
                correct += 1
        print(f'Correct: {correct}/{len(y_test)}')
        
def save_model(model, file_name='my_iris_model.pt'):
    torch.save(model.state_dict(), file_name)

def load_model(file_name='my_iris_model.pt'):
    model = Model()
    model.load_state_dict(torch.load(file_name))
    model.eval()
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data('dataset/iris.csv')
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
