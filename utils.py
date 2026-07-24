import matplotlib.pyplot as plt

def plot_prediction(Y, predictions):
    plt.plot(Y, label="True data")
    plt.plot(predictions, label="Prediction")

    plt.legend()
    plt.savefig("images/aapl_prediction.png")
    plt.show()

def plot_loss(losses):
    plt.plot(losses, label="Losses")

    plt.savefig("images/losses.png")
    plt.show()