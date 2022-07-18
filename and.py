from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np


def main(data, ETA, EPOCHS, filename, plot_filename):

    df = pd.DataFrame(AND)

    X,y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename)
    save_plot(df, plot_filename, model)


if __name__ == '__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    main(data = AND, ETA = 0.3, EPOCHS = 10, filename = 'and.model', plot_filename = "and.png")