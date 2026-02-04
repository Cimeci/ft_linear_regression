from train import Train
from predict import Predict
import sys
import os

class LinearRegressionModel:
    def __init__(self):
        self.trainer = None
        self.predictor = None
        self.menu = """
================================
|   Linear Regression Model     |
================================
1. Train the model
2. Predict price
3. Visualize training results
4. Calculate precision
5. Clean data
6. More info
7. Clean terminal
0. Exit
"""


    def printTitle(self):
        os.system('clear')
        print(self.menu)

    def askNumber(self):
        while True:
            try:
                nb = int(input("What do you want to do? (Number): "))
                if nb == 1:
                    self.train_model()
                elif nb == 2:
                    self.predict_price()
                elif nb == 3:
                    self.visualize_train()
                elif nb == 4:
                    self.calculate_precision()
                elif nb == 5:
                    self.clean_data()
                elif nb == 6:
                    self.more_info()
                elif nb == 7:
                    self.printTitle()
                elif nb == 0:
                    sys.exit(0)
                else:
                    print("Invalid choice. Please enter a number between 0 and 7.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def train_model(self):
        self.trainer = Train()
        self.trainer.get_gradient()
        self.trainer.normalize_theta()
        self.trainer.save_theta()
        print(f"Training completed. Model parameters saved to 'theta.csv'.\n")

    def predict_price(self):
        self.predictor = Predict()
        self.predictor.load_theta()
        self.predictor.get_km_input()
        price = self.predictor.predict_price(self.predictor.km_input)
        print(f"Estimated price: {price:.2f} €\n")

    def visualize_train(self):
        if self.trainer is None:
            return print("Please train the model first.\n")
        self.trainer.get_line()
        self.trainer.calculate_residuals()
        self.trainer.visualize()

    def calculate_precision(self):
        if self.trainer is None:
            return print("Please train the model first.\n")
        result = self.trainer.calculate_precision()
        print(f"Model precision: {result * 100:.2f}%\n")

    def clean_data(self):
        if os.path.exists("theta.csv"):
            os.remove("theta.csv")
            print("theta.csv has been deleted.\n")
        else:
            print("theta.csv does not exist. No file to delete.\n")

    def more_info(self):
        info = """
This Linear Regression Model allows you to train a model based on mileage data to predict car prices.
You can train the model, make predictions, visualize training results, and calculate the model's precision.

@ 2026 Ft_Linear Regression Model by Ilan (github.com/Cimeci)
"""
        print(info)

if __name__ == "__main__":
    model = LinearRegressionModel()
    while True:
        model.printTitle()
        model.askNumber()