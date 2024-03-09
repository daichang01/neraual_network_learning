from model import model
import torch

def predict_image(image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        return output.argmax(dim=1, keepdim=True)

if __name__ == "__main__":
    from imports_and_data import test_data
    import matplotlib.pyplot as plt

    image, label = test_data[4143]
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.show()
    prediction = predict_image(image.view(1, 1, 28, 28))
    print(f'Predicted Label: {prediction.item()}, Actual Label: {label}')
