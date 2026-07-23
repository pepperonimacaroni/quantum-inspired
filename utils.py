import matplotlib.pyplot as plt

def plot_prediction(Y, predictions):
    plt.plot(Y, label="True data")
    plt.plot(predictions, label="Prediction")

    plt.legend()
    plt.savefig("images/sine_prediction.png")
    plt.show()