import pandas as pd
import matplotlib.pyplot as plt

class Train :

    def __init__(self):
        self.df = self.load_data()
        self.km = self.df.iloc[:, 0]       # feature
        self.price = self.df.iloc[:, 1]    # target
        self.m = len(self.km)

        self.km_mean = self.km.mean()
        self.km_std = self.km.std()
        self.km_norm = (self.km - self.km_mean) / self.km_std

        self.theta0 = 0.0
        self.theta1 = 0.0
        self.theta0_real = 0.0
        self.theta1_real = 0.0

        self.learning_rate = 0.01
        self.n_iterations = 1000

        self.y_line = 0.0
        self.residuals = 0.0
        self.mean_residual = 0.0
        self.std_residual = 0.0

        self.mean_x = 0.0
        self.mean_y = 0.0
        

    def load_data(self):
        try:
            datafile = input("CSV datafile Path (default: data.csv) : ")
            if datafile.strip() == "":
                datafile = "data.csv"
            data = pd.read_csv(datafile)
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")
            return
        return data

    # Gradient descent implementation (gradient = vector of partial derivatives)
    def get_gradient(self):
        for _ in range(self.n_iterations):
            self.estimate = self.theta0 + self.theta1 * self.km_norm # hypothesis function
            error = self.estimate - self.price # error vector (residuals)

            tmp_theta0 = self.learning_rate * (1/self.m) * error.sum() # height adjustment
            tmp_theta1 = self.learning_rate * (1/self.m) * (error * self.km_norm).sum() # slope adjustment 
            
            self.theta0 -= tmp_theta0                  
            self.theta1 -= tmp_theta1

    def normalize_theta(self):
        self.theta1_real = self.theta1 / self.km_std
        self.theta0_real = self.theta0 - self.theta1_real * self.km_mean

    def save_theta(self):
        with open("theta.csv", "w") as f:
            f.write(f"{self.theta0_real},{self.theta1_real}\n")

    def get_line(self):
        self.y_line = self.theta0_real + self.theta1_real * self.km

    def calculate_residuals(self):
        self.residuals = self.price - self.y_line
        self.mean_residual = self.residuals.mean()
        self.std_residual = self.residuals.std()
    
    def print_results(self):
        print("Training completed")
        print("theta0 =", self.theta0_real)
        print("theta1 =", self.theta1_real)
        print(f"Mean error (residual mean) : {self.mean_residual:.4f}")
        print(f"Standard deviation of errors (residual std) : {self.std_residual:.4f}")

    def visualize(self):
        plt.scatter(self.km, self.price, color='blue', label='Data')
        plt.plot(self.km, self.y_line, color='red', label='Regression line')

        # Display the mean point (center)
        mean_x = self.km_mean
        mean_y = self.theta0_real + self.theta1_real * mean_x
        plt.scatter([mean_x], [mean_y], color='green', label='Mean point', zorder=5)

        # Display residuals (vertical lines)
        for x_i, y_i in zip(self.km, self.price):
            y_pred = self.theta0_real + self.theta1_real * x_i
            plt.vlines(x_i, y_pred, y_i, color='gray', linestyle='dotted')

        plt.xlabel("Mileage")
        plt.ylabel("Price")
        plt.title("Linear Regression")
        plt.legend()
        plt.show()

    def calculate_precision(self):
        y = self.price
        y_hat = self.estimate

        mean_y = sum(y) / len(y)

        ss_res = 0.0
        ss_tot = 0.0
        i = 0

        while i < len(y):
            ss_res += (y[i] - y_hat[i]) ** 2
            ss_tot += (y[i] - mean_y) ** 2
            i += 1

        if ss_tot == 0:
            print("Precision: 0.00 (no variance)")
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return r2