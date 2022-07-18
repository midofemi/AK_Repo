from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np

def main(data, ETA, EPOCHS, filename, plot_filename):

    df = pd.DataFrame(OR)

    X,y = prepare_data(df)

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    model_OR = Perceptron(eta=ETA, epochs=EPOCHS)
    model_OR.fit(X, y)

    _ = model_OR.total_loss()
    
    save_model(model_OR, filename)
    save_plot(df, plot_filename, model_OR)


if __name__ == '__main__':
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }
    main(data = OR, ETA = 0.3, EPOCHS = 10, filename = 'or.model', plot_filename = "or.png")