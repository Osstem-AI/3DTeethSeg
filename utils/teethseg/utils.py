import numpy as np

def distance_equation(point_1, point_2):
    distance = np.sqrt(np.sum((point_1-point_2) ** 2, axis=0))
    return distance

def ellipse_equation(x, y, radius):
        eli = (x*x)/((0.95*radius)*(0.95*radius)) + (y*y)/((1.2*radius)*(1.2*radius))
        return eli