# The optimal values of m and b can be actually calculated with way less effort than doing a linear regression.
# this is just to demonstrate gradient descent
from numpy import *


# y = mx + b
# m is slope, b is y-intercept
# calculating MSE = 1 / n * sum(y - (mx + b))
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x, y = points[i, 0], points[i, 1]
        totalError += (y - (m * x + b)) ** 2

    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient, m_gradient = 0, 0
    N = float(len(points))
    for i in range(0, len(points)):
        x, y = points[i, 0], points[i, 1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b, m = starting_b, starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]


def main():
    points = genfromtxt("data/test_score_vs_hours_of_study.csv", delimiter=",")
    # hyper parameter - how fast our model learn
    learning_rate = 0.0001
    # y = mx + b (slope formula)
    initial_b, initial_m = 0, 0  # initial y-intercept guess and initial slope guess

    # number of iteration normal for 200 observations
    num_iterations = 1000
    error = compute_error_for_line_given_points(initial_b, initial_m, points)
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, error))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    error = compute_error_for_line_given_points(b, m, points)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, error))
    return b, m


def get_hour_by_score(b, m):
    needed_score = int(input("Enter needed score (0 - 100): "))
    hours_of_study = m * needed_score + b
    print("If you want to get: " + str(needed_score) + " score, you have to spend " + str(hours_of_study) + " hours of study")


if __name__ == '__main__':
    b, m = main()
    # uncomment following line in case you want to get needed hours of study by score you typed
    # get_hour_by_score(b, m)
