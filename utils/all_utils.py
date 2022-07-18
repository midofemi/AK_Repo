import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib # FOR SAVING MY MODEL AS A BINARY FILE
from matplotlib.colors import ListedColormap #This customize (color) your plot
plt.style.use("fivethirtyeight") # THIS IS STYLE OF GRAPHS
import os
import logging


#This is where our functions will be registered to use. This py file just register helper functions
#These are function you are familiar with Amidu. It just separate your X and Y data just like you do in linear regression

#To generate a doc string generator so you can comment out your code. Just Click on extention on the right hand side, search for 
#"Doc String Generator" and then install
#Once you start doing the comment quotation on any of these helper function, you'll be able to generate a doc string template
#These comment are important so other people can see what you did on those functions

def prepare_data(df):
  """It is use to separate the dependent variable and independent features

  Args:
      df (DataFrame): Pandas Dataset

  Returns:
      tuple: It returns the tuples of dependent and independent variables
  """
  X = df.drop("y", axis=1)

  y = df["y"]

  return X, y

def save_model(model, filename):
  """This saves the trained model to a binary file

  Args:
      model (Python Object): Trained model to
      filename (Str): Path to save the trained model
  """
  logging.info("Saving the trained model")
  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  filePath = os.path.join(model_dir, filename) # model/filename
  joblib.dump(model, filePath)
  logging.info(f"Saved the trained model at {filePath}")

def save_plot(df, file_name, model):
  def _create_base_plot(df):
    logging.info("Creating the base plot")
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02):
    logging.info("Plotting the decision region")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values # as a array
    x1 = X[:, 0] 
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1 , x1.max() + 1
    x2_min, x2_max = x2.min() -1 , x2.max() + 1  

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()



  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)
  logging.info(f"Saving the plot at {plotPath}")