import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

# Ignore covariance warnings from curve_fit
warnings.filterwarnings('ignore', message='Covariance of the parameters could not be estimated', category=OptimizeWarning)

# Turn all other RuntimeWarnings into errors
warnings.filterwarnings('error', category=RuntimeWarning)

# Use RMSE instead of MSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Do different types of regression on the dataset to find the best fit line
class InitialAnalysis:
    def __init__(self, data):
        # Create a dict of models (best fit lines according to RMSE)
        self.models = {}
        self.corr = str()

        # Transformation analysis
        # self.trasnformationAnalysis(data=data)

        # Multi regression
        self.multiRegressions(data=data)
    
    def trasnformationAnalysis(self, data):
        # Get dimensions of the dataset (whether multi-dimensional x or not)
        dimensions = data.shape[1] - 1

        # Perform Exploratory Data Analysis
        analysis = data.describe().to_string()

        # Correlation between all x's and y
        for i in range(dimensions):
            analysis += "\n\nCorrelation between x" + str(i+1) + " and y:\n"
            analysis += data["x" + str(i+1)].corr(data["y"]).astype(str)

        # analysis += "\n\nCorrelation between x and y:\n"
        # analysis += data["x"].corr(data["y"]).astype(str)

        # Transformations to all dimensions of x
        analysis += "\n\nLog transformations of x:"
        for i in range(dimensions):
            try:
                analysis += "\n\nLog transformation of x" + str(i+1) + ": \n"
                analysis += data["x" + str(i+1)].apply(np.log).describe().to_string()
                analysis += "\nCorrelation between log(x" + str(i+1) + ") and y:"
                analysis += data["x" + str(i+1)].apply(np.log).corr(data["y"]).astype(str)
            except:
                analysis += ("\n\nLog transformation of x" + str(i+1) + " should not be "
                            "considered as it contains negative values.")

        analysis += "\n\nSquare root transformations of x:"
        for i in range(dimensions):
            try:
                analysis += "\n\nSquare root transformation of x" + str(i+1) + ": \n"
                analysis += data["x" + str(i+1)].apply(np.sqrt).describe().to_string()
                analysis += "\nCorrelation between sqrt(x" + str(i+1) + ") and y:"
                analysis += data["x" + str(i+1)].apply(np.sqrt).corr(data["y"]).astype(str)
            except:
                analysis += ("\n\nSquare root transformation of x" + str(i+1) + " should not be "
                            "considered as it contains negative values.")

        analysis += "\n\nSquare transformations of x:"
        for i in range(dimensions):
            analysis += "\n\nSquare transformation of x" + str(i+1) + ": \n"
            analysis += data["x" + str(i+1)].apply(np.square).describe().to_string()
            analysis += "\nCorrelation between square(x" + str(i+1) + ") and y:"
            analysis += data["x" + str(i+1)].apply(np.square).corr(data["y"]).astype(str)

        analysis += "\n\nCube transformations of x:"
        for i in range(dimensions):
            analysis += "\n\nCube transformation of x" + str(i+1) + ": \n"
            analysis += data["x" + str(i+1)].apply(np.power, args=(3,)).describe().to_string()
            analysis += "\nCorrelation between cube(x" + str(i+1) + ") and y:"
            analysis += data["x" + str(i+1)].apply(np.power, args=(3,)).corr(data["y"]).astype(str)

        analysis += "\n\nExponential transformations of x:"
        for i in range(dimensions):
            try:
                analysis += "\n\nExponential transformation of x" + str(i+1) + ": \n"
                analysis += data["x" + str(i+1)].apply(np.exp).describe().to_string()
                analysis += "\nCorrelation between exp(x" + str(i+1) + ") and y:"
                analysis += data["x" + str(i+1)].apply(np.exp).corr(data["y"]).astype(str)
            except:
                analysis += ("\n\nExponential transformation of x" + str(i+1) + " should not be "
                            "considered as it contains very large values")

        analysis += "\n\nSine transformations of x:"
        for i in range(dimensions):
            analysis += "\n\nSine transformation of x" + str(i+1) + ": \n"
            analysis += data["x" + str(i+1)].apply(np.sin).describe().to_string()
            analysis += "\nCorrelation between sin(x" + str(i+1) + ") and y:"
            analysis += data["x" + str(i+1)].apply(np.sin).corr(data["y"]).astype(str)

        analysis += "\n\nCosine transformations of x:"
        for i in range(dimensions):
            analysis += "\n\nCosine transformation of x" + str(i+1) + ": \n"
            analysis += data["x" + str(i+1)].apply(np.cos).describe().to_string()
            analysis += "\nCorrelation between cos(x" + str(i+1) + ") and y:"
            analysis += data["x" + str(i+1)].apply(np.cos).corr(data["y"]).astype(str)

        self.corr = analysis

    # For debugging purposes
    # print(perform_analysis(data))

    def multiRegressions(self, data):
        # Fit a linear model to the data
        linear_model = LinearRegression()

        # Fit the model and add the linear model fitness to the dict of models
        try:
                linear_model.fit(data.drop("y", axis=1), data["y"])
                y_pred = linear_model.predict(data.drop("y", axis=1))
                self.models.update({"Linear model": rmse(data["y"], y_pred)})
                # self.models.append(("Linear model", rmse(data["y"], y_pred)))
        except:
            self.models.update({"Linear model": 500.0})
            # self.models.append(("Linear model", 500.0))

        # Optionally, test predictions on the test dataset
        # y_pred = linear_model.predict(test_data.drop("y", axis=1))

        def exponential_func(x, *params):
            num_dims = x.shape[1]  # Number of dimensions in the data
            y = np.zeros(x.shape[0])  # Initialize the output array

            for i in range(num_dims):
                # Assuming the first parameter for each dimension is the scale and the second is the rate
                scale, rate = params[i*2:(i+1)*2]
                y += scale * np.exp(-rate * x.iloc[:, i])

            return y

        try:
            # Set initial guesses for each dimension: [scale1, rate1, scale2, rate2, ...]
            initial_guess = [1.0, 1.0] * data.shape[1] 
            popt, pcov = curve_fit(exponential_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
            y_pred = exponential_func(data.drop("y", axis=1), *popt)
            self.models.update({"Exponential model": rmse(data["y"], y_pred)})
        except Exception as e:
            self.models.update({"Exponential model": 500.0})
            print(e)

        # Optionally, test predictions on the test dataset
        # y_pred = exponential_func(test_data["x"], *popt)
            
        # Fit a sinusoidal model to the data
        def sinusoidal_func(x, *params):
            # Assuming params are ordered as [a1, b1, c1, d1, a2, b2, c2, d2, ...] for each dimension
            num_dims = x.shape[1]
            y = np.zeros(x.shape[0])

            for i in range(num_dims):
                a, b, c, d = params[i*4:(i+1)*4]
                y += a * np.sin(b * x.iloc[:, i] + c) + d

            return y

        try:
            initial_guess = [1, 1e-6, 1, 1] * data.shape[1]
            popt, pcov = curve_fit(sinusoidal_func, data.drop("y", axis=1), data["y"], initial_guess, maxfev=10000)
            self.models.update({"Sinusoidal model": rmse(data["y"], sinusoidal_func(data.drop("y", axis=1), *popt))})
        except Exception as e:
            self.models.update({"Sinusoidal model": 500.0})
            print(e)

        # Optionally, test predictions on the test dataset
        # y_pred = sinusoidal_func(test_data["x"], *popt)
            
        # Fit a cosine model to the data
        def cosine_func(x, *params):
            num_dims = x.shape[1]  # Number of dimensions in the data
            y = np.zeros(x.shape[0])  # Initialize the output array

            for i in range(num_dims):
                # Assuming params are ordered as [a1, b1, c1, d1, a2, b2, c2, d2, ...] for each dimension
                a, b, c, d = params[i*4:(i+1)*4]
                y += a * np.cos(b * x.iloc[:, i] + c) + d

            return y

        try:
            # Set initial guesses for each dimension: [a1, b1, c1, d1, a2, b2, c2, d2, ...]
            initial_guess = [1, 1e-6, 1, 1] * data.shape[1]
            popt, pcov = curve_fit(cosine_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
            y_pred = cosine_func(data.drop("y", axis=1), *popt)
            self.models.update({"Cosine model": rmse(data["y"], y_pred)})
        except Exception as e:
            self.models.update({"Cosine model": 500.0})
            print(e)
            
        # Optionally, test predictions on the test dataset
        # y_pred = cosine_func(test_data["x"], *popt)
            
        # Fit a square root model to the data
        def square_root_func(x, *params):
            num_dims = x.shape[1]  # Number of dimensions in the data
            y = np.zeros(x.shape[0])  # Initialize the output array

            for i in range(num_dims):
                # Assuming params are ordered as [a1, b1, c1, a2, b2, c2, ...] for each dimension
                a, b, c = params[i*3:(i+1)*3]
                y += a * np.sqrt(b * x.iloc[:, i]) + c

            return y

        try:
            # Set initial guesses for each dimension: [a1, b1, c1, a2, b2, c2, ...]
            initial_guess = [1, 1e-6, 1] * data.shape[1]
            popt, pcov = curve_fit(square_root_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
            y_pred = square_root_func(data.drop("y", axis=1), *popt)
            self.models.update({"Square root model": rmse(data["y"], y_pred)})
        except Exception as e:
            self.models.update({"Square root model": 500.0})
            print(e)
                
        # Optionally, test predictions on the test dataset
        # y_pred = square_root_func(test_data["x"], *popt)
            
        # Fit a cubic model to the data
        def cubic_func(x, *params):
            num_dims = x.shape[1]  # Number of dimensions in the data
            y = np.zeros(x.shape[0])  # Initialize the output array

            for i in range(num_dims):
                # Assuming params are ordered as [a1, b1, c1, d1, a2, b2, c2, d2, ...] for each dimension
                a, b, c, d = params[i*4:(i+1)*4]
                y += a * x.iloc[:, i]**3 + b * x.iloc[:, i]**2 + c * x.iloc[:, i] + d

            return y

        try:
            # Set initial guesses for each dimension: [a1, b1, c1, d1, a2, b2, c2, d2, ...]
            initial_guess = [1, 1e-6, 1, 1] * data.shape[1]
            popt, pcov = curve_fit(cubic_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
            y_pred = cubic_func(data.drop("y", axis=1), *popt)
            self.models.update({"Cubic model": rmse(data["y"], y_pred)})
        except Exception as e:
            self.models.update({"Cubic model": 500.0})
            print(e)

        # Plot the regression line and convert it to an image
        # im = plot_regression(data, cubic_func, popt, rmse(data["y"], cubic_func(data["x"], *popt)))

        # Optionally, test predictions on the test dataset
        # y_pred = cubic_func(test_data["x"], *popt)
            
        # Fit a hyperbolic model to the data
        def hyperbolic_func(x, *params):
            num_dims = x.shape[1]  # Number of dimensions in the data
            y = np.zeros(x.shape[0])  # Initialize the output array

            for i in range(num_dims):
                # Assuming params are ordered as [a1, b1, c1, a2, b2, c2, ...] for each dimension
                a, b, c = params[i*3:(i+1)*3]
                y += a / (b * x.iloc[:, i] + c)

            return y

        try:
                # Set initial guesses for each dimension: [a1, b1, c1, a2, b2, c2, ...]
                initial_guess = [1, 1e-6, 1] * data.shape[1]
                popt, pcov = curve_fit(hyperbolic_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
                y_pred = hyperbolic_func(data.drop("y", axis=1), *popt)
                self.models.update({"Hyperbolic model": rmse(data["y"], y_pred)})
        except Exception as e:
            self.models.update({"Hyperbolic model": 500.0})
            print(e)


        # Optionally, test predictions on the test dataset
        # y_pred = hyperbolic_func(test_data["x"], *popt)
            
        # Fit a ln model to the data
        def ln_func(x, *params):
            num_dims = x.shape[1]  # Number of dimensions in the data
            y = np.zeros(x.shape[0])  # Initialize the output array

            for i in range(num_dims):
                # Assuming params are ordered as [a1, b1, c1, a2, b2, c2, ...] for each dimension
                a, b, c = params[i*3:(i+1)*3]
                y += a * np.log(b * x.iloc[:, i] + c)  # Using natural logarithm (ln)

            return y

        try:
            # Set initial guesses for each dimension: [a1, b1, c1, a2, b2, c2, ...]
            initial_guess = [1, 1e-6, 1] * data.shape[1]
            popt, pcov = curve_fit(ln_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
            y_pred = ln_func(data.drop("y", axis=1), *popt)
            self.models.update({"Ln model": rmse(data["y"], y_pred)})
        except Exception as e:
            self.models.update({"Ln model": 500.0})
            print(e)

        # Fit a tan model to the data
        def tan_func(x, *params):
            num_dims = x.shape[1]  # Number of dimensions in the data
            y = np.zeros(x.shape[0])  # Initialize the output array

            for i in range(num_dims):
                # Assuming params are ordered as [a1, b1, c1, d1, a2, b2, c2, d2, ...] for each dimension
                a, b, c, d = params[i*4:(i+1)*4]
                y += a * np.tan(b * x.iloc[:, i] + c) + d

            return y

        try:
            # Set initial guesses for each dimension: [a1, b1, c1, d1, a2, b2, c2, d2, ...]
            initial_guess = [1, 1e-6, 1, 1] * data.shape[1]
            popt, pcov = curve_fit(tan_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
            y_pred = tan_func(data.drop("y", axis=1), *popt)
            self.models.update({"Tan model": rmse(data["y"], y_pred)})
        except Exception as e:
            self.models.update({"Tan model": 500.0})
            print(e)

        # Fit hyperbolic sine model to the data
        # def sinh_func(x, *params):
        #     num_dims = x.shape[1]  # Number of dimensions in the data
        #     y = np.zeros(x.shape[0])  # Initialize the output array

        #     for i in range(num_dims):
        #         # Assuming params are ordered as [a1, b1, c1, d1, a2, b2, c2, d2, ...] for each dimension
        #         a, b, c, d = params[i*4:(i+1)*4]
        #         y += a * np.sinh(b * x.iloc[:, i] + c) + d

        #     return y

        # try:
        #     # Set initial guesses for each dimension: [a1, b1, c1, d1, a2, b2, c2, d2, ...]
        #     initial_guess = [1, 1e-6, 1, 1] * data.shape[1]
        #     popt, pcov = curve_fit(sinh_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
        #     y_pred = sinh_func(data.drop("y", axis=1), *popt)
        #     self.models.update({"Sinh model": rmse(data["y"], y_pred)})
        # except Exception as e:
        #     self.models.update({"Sinh model": 500.0})
        #     print(e)

        # # Fit hyperbolic cosine model to the data
        # def cosh_func(x, *params):
        #     num_dims = x.shape[1]  # Number of dimensions in the data
        #     y = np.zeros(x.shape[0])  # Initialize the output array

        #     for i in range(num_dims):
        #         # Assuming params are ordered as [a1, b1, c1, d1, a2, b2, c2, d2, ...] for each dimension
        #         a, b, c, d = params[i*4:(i+1)*4]
        #         y += a * np.cosh(b * x.iloc[:, i] + c) + d

        #     return y

        # try:
        #     # Set initial guesses for each dimension: [a1, b1, c1, d1, a2, b2, c2, d2, ...]
        #     initial_guess = [1, 1e-6, 1, 1] * data.shape[1]
        #     popt, pcov = curve_fit(cosh_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
        #     y_pred = cosh_func(data.drop("y", axis=1), *popt)
        #     self.models.update({"Cosh model": rmse(data["y"], y_pred)})
        # except Exception as e:
        #     self.models.update({"Cosh model": 500.0})
        #     print(e)

        # # Fit a hyperbolic tangent model to the data
        # def tanh_func(x, *params):
        #     num_dims = x.shape[1]  # Number of dimensions in the data
        #     y = np.zeros(x.shape[0])

        #     for i in range(num_dims):
        #         # Assuming params are ordered as [a1, b1, c1, d1, a2, b2, c2, d2, ...] for each dimension
        #         a, b, c, d = params[i*4:(i+1)*4]
        #         y += a * np.tanh(b * x.iloc[:, i] + c) + d

        #     return y

        # try:
        #     # Set initial guesses for each dimension: [a1, b1, c1, d1, a2, b2, c2, d2, ...]
        #     initial_guess = [1, 1e-6, 1, 1] * data.shape[1]
        #     popt, pcov = curve_fit(tanh_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
        #     y_pred = tanh_func(data.drop("y", axis=1), *popt)
        #     self.models.update({"Tanh model": rmse(data["y"], y_pred)})
        # except Exception as e:
        #     self.models.update({"Tanh model": 500.0})
        #     print(e)

        # # Fit a arcsine model to the data
        # def arcsin_func(x, *params):
        #     num_dims = x.shape[1]  # Number of dimensions in the data
        #     y = np.zeros(x.shape[0])

        #     for i in range(num_dims):
        #         # Assuming params are ordered as [a1, b1, c1, d1, a2, b2, c2, d2, ...] for each dimension
        #         a, b, c, d = params[i*4:(i+1)*4]
        #         y += a * np.arcsin(b * x.iloc[:, i] + c) + d

        #     return y

        # try:
        #     # Set initial guesses for each dimension: [a1, b1, c1, d1, a2, b2, c2, d2, ...]
        #     initial_guess = [1, 1, 1, 1] * data.shape[1]
        #     popt, pcov = curve_fit(arcsin_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
        #     y_pred = arcsin_func(data.drop("y", axis=1), *popt)
        #     self.models.update({"Arcsin model": rmse(data["y"], y_pred)})
        # except Exception as e:
        #     self.models.update({"Arcsin model": 500.0})
        #     print(e)

        # # Fit a acos model to the data
        # def arccos_func(x, *params):
        #     num_dims = x.shape[1]  # Number of dimensions in the data
        #     y = np.zeros(x.shape[0])

        #     for i in range(num_dims):
        #         # Assuming params are ordered as [a1, b1, c1, d1, a2, b2, c2, d2, ...] for each dimension
        #         a, b, c, d = params[i*4:(i+1)*4]
        #         y += a * np.arccos(b * x.iloc[:, i] + c) + d

        #     return y

        # try:
        #     # Set initial guesses for each dimension: [a1, b1, c1, d1, a2, b2, c2, d2, ...]
        #     initial_guess = [-1, 1, 1, 1] * data.shape[1]
        #     popt, pcov = curve_fit(arccos_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
        #     y_pred = arccos_func(data.drop("y", axis=1), *popt)
        #     self.models.update({"Arccos model": rmse(data["y"], y_pred)})
        # except Exception as e:
        #     self.models.update({"Arccos model": 500.0})
        #     print(e)

        # # Fit a arctan model to the data
        # def arctan_func(x, *params):
        #     num_dims = x.shape[1]  # Number of dimensions in the data
        #     y = np.zeros(x.shape[0])

        #     for i in range(num_dims):
        #         # Assuming params are ordered as [a1, b1, c1, d1, a2, b2, c2, d2, ...] for each dimension
        #         a, b, c, d = params[i*4:(i+1)*4]
        #         y += a * np.arctan(b * x.iloc[:, i] + c) + d

        #     return y

        # try:
        #     # Set initial guesses for each dimension: [a1, b1, c1, d1, a2, b2, c2, d2, ...] 
        #     initial_guess = [1, 1, 1, 1] * data.shape[1]
        #     popt, pcov = curve_fit(arctan_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
        #     y_pred = arctan_func(data.drop("y", axis=1), *popt)
        #     self.models.update({"Arctan model": rmse(data["y"], y_pred)})
        # except Exception as e:
        #     self.models.update({"Arctan model": 500.0})
        #     print(e)

        # Fit a absolute value model to the data
        # def abs_func(x, *params):
        #     num_dims = x.shape[1]  # Number of dimensions in the data
        #     y = np.zeros(x.shape[0])

        #     for i in range(num_dims):
        #         # Assuming params are ordered as [a1, b1, a2, b2, ...] for each dimension
        #         a, b = params[i*2:(i+1)*2]
        #         y += a * np.abs(b * x.iloc[:, i])

        #     return y

        # try:
        #     # Set initial guesses for each dimension: [a1, b1, a2, b2, ...] 
        #     initial_guess = [1, 1] * data.shape[1]
        #     popt, pcov = curve_fit(abs_func, data.drop("y", axis=1), data["y"], p0=initial_guess, maxfev=10000)
        #     y_pred = abs_func(data.drop("y", axis=1), *popt)
        #     self.models.update({"Absolute value model": rmse(data["y"], y_pred)})
        # except Exception as e:
        #     self.models.update({"Absolute value model": 500.0})
        #     print(e)