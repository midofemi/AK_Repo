from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

def main(data, ETA, EPOCHS, filename, plot_filename):

    df = pd.DataFrame(AND)
    logging.info(f"This is the actual dataframe{df}")
    X,y = prepare_data(df)

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename)
    #save_plot(df, plot_filename, model)



if __name__ == '__main__':
    # AND = {
    #     "x1": [0,0,1,1],
    #     "x2": [0,1,0,1],
    #     "y": [0,0,0,1],
    # }
    AND = pd.read_csv('C:/Users/midof/OneDrive/Documents/Dataset/Flowers.csv')
    try:
        logging.info(">>>>>>>> Starting training >>>>>>>")
        main(data = AND, ETA = 0.3, EPOCHS = 10, filename = 'and.model', plot_filename = "and.png")
        logging.info("<<<<<<<< Training Done Sucessfully <<<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
