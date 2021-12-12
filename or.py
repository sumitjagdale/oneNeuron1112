from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s:%(levelname)s:%(module)s:%(message)s]"
log_dir="logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=str(logging_str), filemode="a")

def main(data, eta, epochs, filename, plotFileName):
    
    
    df = pd.DataFrame(OR)
    logging.info(f"This is the actual dataframe:{df}")

    X,y = prepare_data(df)

    model_OR = Perceptron(eta=ETA, epochs=EPOCHS)
    model_OR.fit(X, y)

    _ = model_OR.total_loss()

    save_model(model_OR, filename=filename)
    save_plot(df, "plotfileName", model_OR)

if __name__ == "__main__":
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }
    ETA =0.3
    EPOCHS =10
    try:
        logging.info(">>>>> starting training >>>>>>")
        main(data=OR, eta=ETA, epochs=EPOCHS, filename="or.model",plotFileName="or.png")
        logging.info("<<<<< training finished <<<<<<\n")
    except Exception as e:
        logging.exceptions(e)
