import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Problem:
    # Define benchmarking functions for GPSR
    # x^4 + x^3 + x^2 + x
    @staticmethod
    def f1(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(-1, 1)
            y[i] = x[i]**4 + x[i]**3 + x[i]**2 + x[i]
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # x^5 - 2x^3 + x
    @staticmethod
    def f2(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(-1, 1)
            y[i] = x[i]**5 - 2*x[i]**3 + x[i]
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # x^6 - 2x^4 + x^2
    @staticmethod
    def f3(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(-1, 1)
            y[i] = x[i]**6 - 2*x[i]**4 + x[i]**2
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # x^3 + x^2 + x
    @staticmethod
    def f4(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(-1, 1)
            y[i] = x[i]**3 + x[i]**2 + x[i]
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # x^5 + x^4 + x^3 + x^2 + x
    @staticmethod
    def f5(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(-1, 1)
            y[i] = x[i]**5 + x[i]**4 + x[i]**3 + x[i]**2 + x[i]
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # x^6 + x^5 + x^4 + x^3 + x^2 + x
    @staticmethod
    def f6(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(-1, 1)
            y[i] = x[i]**6 + x[i]**5 + x[i]**4 + x[i]**3 + x[i]**2 + x[i]
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # sin(x^2) * cos(x) - 1
    @staticmethod
    def f7(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(-1, 1)
            y[i] = np.sin(x[i]**2) * np.cos(x[i]) - 1
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # sin(x) + sin(x + x^2)
    @staticmethod
    def f8(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(-1, 1)
            y[i] = np.sin(x[i]) + np.sin(x[i] + x[i]**2)
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # ln(x + 1) + ln(x^2 + 1)
    # @staticmethod 
    def f9(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(0, 2)
            y[i] = np.log(x[i] + 1) + np.log(x[i]**2 + 1)
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # sqrt(x)
    @staticmethod
    def f10(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(0, 4)
            y[i] = np.sqrt(x[i])
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # sin(x1) + sin(x2^2)
    @staticmethod
    def f11(seed, train=1000, test=200):
        samples = train + test
        x = np.zeros((samples, 2))
        y = np.zeros(samples)
        for i in range(samples):
            for j in range(2):
                x[i, j] = random.uniform(0, 1)
            y[i] = np.sin(x[i, 0]) + np.sin(x[i, 1]**2)
        data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # 2*sin(x1) + cos(x2)
    @staticmethod
    def f12(seed, train=1000, test=200):
        samples = train + test
        x = np.zeros((samples, 2))
        y = np.zeros(samples)
        for i in range(samples):
            for j in range(2):
                x[i, j] = random.uniform(0, 1)
            y[i] = 2*np.sin(x[i, 0]) + np.cos(x[i, 1])
        data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # 10 * (5 + (x1-3**2 + (x2-3)^2 + (x3-3)**2) + (x4-3)**2 + (x5-3)**2)**-1
    @staticmethod
    def f13(seed, train=2000, test=400):
        samples = train + test
        x = np.zeros((samples, 5)) # 5-dimensional
        y = np.zeros(samples)
        for i in range(samples):
            for j in range(5):
                x[i, j] = random.uniform(0.05, 6.05)
            y[i] = 10 * (5 + (x[i, 0]-3)**2 + (x[i, 1]-3)**2 + (x[i, 2]-3)**2 + (x[i, 3]-3)**2 + (x[i, 4]-3)**2)**-1
        data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'x3': x[:, 2], 'x4': x[:, 3], 'x5': x[:, 4], 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # 30 * ((x1 - 1)*(x2 - 1))/(x3^2 * (x1 - 10))
    @staticmethod
    def f14(seed, train=1000, test=200):
        samples = train + test
        x = np.zeros((samples, 3))
        y = np.zeros(samples)
        for i in range(samples):
            for j in range(3):
                if j == 0:
                    x[i, j] = random.uniform(0.05, 2.0)
                if j == 1:
                    x[i, j] = random.uniform(1.0, 2.0)
                if j == 2:
                    x[i, j] = random.uniform(0.05, 2.0)
            y[i] = 30 * ((x[i, 0] - 1)*(x[i, 2] - 1))/(x[i, 1]**2 * (x[i, 0] - 10))
        data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'x3': x[:, 2], 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # Keijzer 1 0.3* x * sin * (2 * pi * x)
    @staticmethod
    def f15(seed, train=20, test=1000):
        x_train = [random.uniform(-1, 1) for _ in range(train)]
        y_train = [0.3 * x * np.sin(x) * (2 * np.pi * x) for x in x_train]
        train_data = pd.DataFrame({'x1': x_train, 'y': y_train})

        x_test = [random.uniform(-1.1, 1.1) for _ in range(test)]
        y_test = [0.3 * x * np.sin(x) * (2 * np.pi * x) for x in x_test]
        test_data = pd.DataFrame({'x1': x_test, 'y': y_test})

        return train_data, test_data

    # Keijzer 14 8.0 / (2 + x1**2 + x2**2)
    @staticmethod
    def f16(seed, train=25, test=100):
        samples = train + test
        x = np.zeros((samples, 2))
        y = np.zeros(samples)
        for i in range(samples):
            for j in range(2):
                x[i, j] = random.uniform(-1.0, 1.0)
            y[i] = 8.0 / (2 + x[i, 0]**2 + x[i, 1]**2)
        data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test
        
    # Modqua function: 4x^4 + 3x^3 + 2x^2 + x
    @staticmethod
    def f20(seed, train=200, test=40):
        samples = train + test
        x = np.zeros(samples)
        y = np.zeros(samples)
        for i in range(samples):
            x[i] = random.uniform(-2, 2)
            y[i] = 4*x[i]**4 + 3*x[i]**3 + 2*x[i]**2 + x[i]
        data = pd.DataFrame({'x1': x, 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # Nonic function: sum(i=1, i=9)(x_i)
    @staticmethod
    def f21(seed, train=1000, test=200):
        samples = train + test
        x = np.zeros((samples, 9))
        y = np.zeros(samples)
        for i in range(samples):
            for j in range(9):
                x[i, j] = random.uniform(-2, 2)
            y[i] = np.sum(x[i, :])
        data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'x3': x[:, 2], 'x4': x[:, 3], 
                            'x5': x[:, 4], 'x6': x[:, 5], 'x7': x[:, 6], 'x8': x[:, 7], 
                            'x9': x[:, 8], 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # Hartman -exp(-sum(i=1, i=4)((x_i)^2)
    @staticmethod
    def f22(seed, train=2000, test=400):
        samples = train + test
        x = np.zeros((samples, 4))
        y = np.zeros(samples)
        for i in range(samples):
            for j in range(4):
                x[i, j] = random.uniform(0, 2)
            y[i] = -np.exp(-np.sum(x[i, :]**2))
        data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'x3': x[:, 2], 'x4': x[:, 3], 'y': y})
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # Feynman equation number I.6.2a: exp(-theta**2/2)/sqrt(2*pi)
    @staticmethod
    def f23(seed, train=200, test=40):
        # Load the dataset from the file
        data = pd.read_csv('../dataset/Feynman_with_units/I.6.2a', delim_whitespace=True, header=None)
        # Select random samples
        data = data.sample(n=train+test)
        # Count the number of columns
        num_columns = data.shape[1]
        # Generate new column names
        # Create a list for column names: ['x1', 'x2', ..., 'y']
        column_names = ['x' + str(i) for i in range(1, num_columns)]
        column_names.append('y')
        # Rename columns in the dataframe
        data.columns = column_names
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # Feynman function number I.9.18: G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    @staticmethod
    def f24(seed, train=2000, test=400):
        # Load the dataset from the file
        data = pd.read_csv('../dataset/Feynman_with_units/I.9.18', delim_whitespace=True, header=None)
        # Select random 1000 rows
        data = data.sample(n=train + test)
        # Count the number of columns
        num_columns = data.shape[1]
        # Generate new column names
        # Create a list for column names: ['x1', 'x2', ..., 'y']
        column_names = ['x' + str(i) for i in range(1, num_columns)]
        column_names.append('y')
        # Rename columns in the dataframe
        data.columns = column_names
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return data

    # Feynman function number I.10.7: m_0/sqrt(1-v**2/c**2)
    @staticmethod
    def f25(seed, train=1000, test=200):
        samples = train + test
        x = np.zeros((samples, 3))
        y = np.zeros(samples)
        for i in range(samples):
            for j in range(3):
                if j == 0:
                    x[i, j] = random.uniform(1.0, 5.0)
                if j == 1:
                    x[i, j] = random.uniform(1.0, 2.0)
                if j == 2:
                    x[i, j] = random.uniform(3.0, 10.0)
            y[i] = x[i, 0]/np.sqrt(1-x[i, 1]**2/x[i, 2]**2)
        data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'x3': x[:, 2], 'y': y})
        train, test = train_test_split(data, test_size=2000, random_state=seed)
        return train, test
        
    # Airfoil data
    @staticmethod
    def f26(seed, train=1127, test=376):
        # # Load the dataset from the file
        data = pd.read_csv('../dataset/airfoil_self_noise.dat', sep="\t", header=None) 
        # Select the appropriate number of samples
        data = data.sample(n=train + test, random_state=seed)
        # Count the number of columns
        num_columns = data.shape[1]
        # Generate new column names
        # Create a list for column names: ['x1', 'x2', ..., 'y']
        column_names = ['x' + str(i) for i in range(1, num_columns)]
        column_names.append('y')
        # Rename columns in the dataframe
        data.columns = column_names
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # BHouse data
    @staticmethod
    def f27(seed, train=380, test=126):
        # Load the dataset
        data = pd.read_csv("../dataset/boston.csv", sep=",", header=0)
        # Sample 380 rows
        data = data.sample(n=train + test, random_state=seed)
        # Count the number of columns
        num_columns = data.shape[1]
        # Generate new column names
        # Create a list for column names: ['x1', 'x2', ..., 'y']
        column_names = ['x' + str(i) for i in range(1, num_columns)]
        column_names.append('y')
        # Rename columns in the dataframe
        data.columns = column_names
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test

    # Tower data
    @staticmethod
    def f28(seed, train=2499, test=1250):
        # Load the dataset
        data = pd.read_csv("../dataset/tower.txt", sep="\t", header=0)
        data = data.sample(n=train + test, random_state=seed)
        # Count the number of columns
        num_columns = data.shape[1]
        # Generate new column names
        # Create a list for column names: ['x1', 'x2', ..., 'y']
        column_names = ['x' + str(i) for i in range(1, num_columns)]
        column_names.append('y')
        # Rename columns in the dataframe
        data.columns = column_names
        train, test = train_test_split(data, test_size=test, random_state=seed)
        return train, test
    
functions = {
    1: Problem.f1,
    2: Problem.f2,
    3: Problem.f3,
    4: Problem.f4,
    5: Problem.f5,
    6: Problem.f6,
    7: Problem.f7,
    8: Problem.f8,
    9: Problem.f9,
    10: Problem.f10,
    11: Problem.f11,
    12: Problem.f12,
    13: Problem.f13,
    14: Problem.f14,
    15: Problem.f15,
    16: Problem.f16,
    20: Problem.f20,
    21: Problem.f21,
    22: Problem.f22,
    26: Problem.f26,
    27: Problem.f27,
    28: Problem.f28
}