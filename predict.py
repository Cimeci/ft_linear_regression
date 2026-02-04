class Predict :
    def __init__(self):
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.km_input = 0.0
        self.estimated_price = 0.0

    def load_theta(self):
        try:
            with open("theta.csv", "r") as f:
                self.theta0, self.theta1 = map(float, f.readline().strip().split(','))
        except FileNotFoundError:
            print("File 'theta.csv' not found. Please train the model first.")
            return

    def get_km_input(self):
        try:
            self.km_input = float(input("Enter the mileage: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            return

    def predict_price(self, km):
        self.estimated_price = self.theta0 + self.theta1 * km
        return self.estimated_price