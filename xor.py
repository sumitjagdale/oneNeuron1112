from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np

def main(data, eta, epochs, filename, plotFileName):
    
    
    df = pd.DataFrame(XOR)
    print(df)

    X,y = prepare_data(df)

    model_XOR = Perceptron(eta=ETA, epochs=EPOCHS)
    model_XOR.fit(X, y)

    _ = model_XOR.total_loss()

    save_model(model_XOR, filename=filename)
    save_plot(df, "plotfileName", model_XOR)

if __name__ == "__main__":

    XOR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y":  [0,1,1,0],
    }
    ETA = 0.3
    EPOCHS = 10
    
    main(data=XOR, eta=ETA, epochs=EPOCHS, filename="xor.model",plotFileName="xor.png")