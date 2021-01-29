# Import required libraries
import numpy as np 

# Import necessary modules
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

class LooTraining:
    
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
        loocv = LeaveOneOut()
        errors = []
        model_list = []
        
        for train_indices, test_indices in loocv.split(x_data):
                #training the data        
                train_X, train_y = x_data[train_indices], y_data[train_indices]
                test_X, test_y = x_data[test_indices], y_data[test_indices]

                #making the model
                model = LinearRegression()
                model = model.fit(train_X, train_y)

                #checking and storing the error
                rmse = np.sqrt(mean_squared_error(test_y, model.predict(test_X)))
                errors.append(rmse)

                #storing a tuple that combines the model and the error
                model_list.append((model,rmse))
    
      
        self.average_error = sqrt(sum(errors) / len(errors)) 
        self.model_tuple = model_list
    
    
    def average_rmse(self):
        return (self.average_error)
    
    def best_model(self):
        smallest = abs(self.average_error - self.model_tuple[0][1]) #default
        for element in self.model_tuple: #element[0] is the model and [1] is his rmse
            if abs(self.average_error - element[1]) < smallest:
                smallest = abs(self.average_error - element[1]) 
                b_model = element[0] #Could say best, but already have best_model name

        return b_model