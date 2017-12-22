# Hanuma Venkata Sai Sudheer, Raja
# 1001-541-257
# 2017-09-17
# Assignment_01_02

import random
import math as m
import matplotlib

matplotlib.use('TkAgg')
# Do not change the order
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk


class DisplayActivationFunctions:
    """Class to train and plot decision boundary using perceptron learning rule"""

    def __init__(self, root, master, *args, **kwargs):
        """Display preferences and default variable initialization.
           init is always called first on object creation"""
        self.master = master
        self.root = root

        #########################################################################
        #  Set up the constants and default values
        #########################################################################

        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10

        # Default weight,activation function and bias values
        self.input_weight1 = 1
        self.input_weight2 = 1
        self.bias = 0
        # Threshold
        self.theta = 0

        # Learning rate
        # self.Learning_rate = 0.05

        self.activation_function = "Symmetrical hard limit"

        # Input array
        self.input_array_x = np.array([0, 0, 0, 0])
        self.input_array_y = np.array([0, 0, 0, 0])

        # Weight array
        self.weight_array = np.array([0, 0])

        # Bias array
        self.bias_array = np.array([0])

        # Target output array
        self.output_array = np.array([1, 1, -1, -1])

        #########################################################################
        #  Set up the plotting area
        #########################################################################

        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input')
        self.axes.set_ylabel('Output')
        self.axes.set_title("Learning visual")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################

        self.sliders_frame = tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(2, weight=2)
        self.sliders_frame.columnconfigure(0, weight=5, uniform='xx')

        # set up sliders
        # Slider 1 -> Weight 1
        self.input_weight1_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="Input Weight 1",
                                             command=lambda event: self.input_weight1_slider_callback())
        self.input_weight1_slider.set(self.input_weight1)
        self.input_weight1_slider.bind("<ButtonRelease-1>", lambda event: self.input_weight1_slider_callback())
        self.input_weight1_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Slider 2 -> Weight 2
        self.input_weight2_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="Input Weight 2",
                                             command=lambda event: self.input_weight2_slider_callback())
        self.input_weight2_slider.set(self.input_weight2)
        self.input_weight2_slider.bind("<ButtonRelease-1>", lambda event: self.input_weight2_slider_callback())
        self.input_weight2_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Slider 3 -> Bias
        self.bias_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Bias",
                                    command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
        self.bias_slider.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for drop down and buttons
        #########################################################################

        self.buttons_frame = tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.buttons_frame.rowconfigure(5, weight=2)
        self.buttons_frame.columnconfigure(0, weight=5, uniform='xx')
        self.label_for_activation_function = tk.Label(self.buttons_frame, text="Activation Function",
                                                      justify="center")
        self.label_for_activation_function.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.activation_function_variable = tk.StringVar()

        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Symmetrical hard limit", "Hyperbolic Tangent", "Linear",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        # Default selection for drop down
        self.activation_function_variable.set("Symmetrical hard limit")
        self.activation_function_dropdown.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())

        # Button 1 label
        self.generate_button_label = tk.Label(self.buttons_frame, text="Generate random data",
                                              justify="center")
        self.generate_button_label.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Button 1
        # Do not use command as generate_input(). Method will not be called correctly when buttons are pressed.
        self.generate_button = tk.Button(self.buttons_frame, text="Generate", height=2,
                                         command=self.generate_random_inputs)
        self.generate_button.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Button 2 label
        self.train_button_label = tk.Label(self.buttons_frame, text="Train",
                                           justify="center")
        self.train_button_label.grid(row=4, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # Button 2
        self.train_button = tk.Button(self.buttons_frame, text="Train", command=self.train_algorithm)
        self.train_button.grid(row=5, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

    def generate_random_inputs(self):
        """Generate random inputs and bias in range (-10, 10)"""
        # from Raja_01_01 import StatusBar
        input_set = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.input_array_x = random.sample(input_set, 4)
        self.input_array_y = random.sample(input_set, 4)

        print(self.input_array_x)
        print(self.input_array_y)

        # Copy to numpy array <-- Weights
        self.weight_array = [self.input_weight1, self.input_weight2]

        # Copy to numpy array <-- Bias
        self.bias_array = [self.bias]

        print(self.weight_array)

        # Single value for bias
        print(self.bias_array)

        # Plot inputs on graph
        self.display_input_points()

    def display_input_points(self):
        """Draw input points on graph"""
        # Axis setup
        self.axes.cla()
        self.axes.cla()
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.xlabel("Input")
        plt.ylabel("Output")
        # Plot input points {Green for +1 class and Red for -1 class}
        #colors = ['green', 'red']
        self.axes.scatter([self.input_array_x[0], self.input_array_x[1]], [self.input_array_y[0], self.input_array_y[1]], s=150, marker='o', color='g', edgecolor='none')
        self.axes.scatter([self.input_array_x[2], self.input_array_x[3]], [self.input_array_y[2], self.input_array_y[3]], s=150, marker='o', color='r', edgecolor='none')
        plt.title(self.activation_function)
        # Display graph on plot
        self.canvas.draw()

    def display_activation_function(self):
        """Draws decision boundary on graph"""
        # Axis setup
        #self.axes.cla()
        #self.axes.cla()
        #self.axes.xaxis.set_visible(True)
        #plt.xlim(self.xmin, self.xmax)
        #plt.ylim(self.ymin, self.ymax)
        #plt.xlabel("Input")
        #plt.ylabel("Output)

        # Plot decision boundary line
        x = np.linspace(-10, 10)
        if self.weight_array[1] == 0:
            y = -(self.bias_array[0] + 0)
        else:
            y = -(self.bias_array[0] + (self.weight_array[0] * x) / self.weight_array[1])
        # Draw line plot
        # self.axes.plot(x, y)
        # Fill color in plot
        self.axes.fill_between(x, y, 10, interpolate=True, color='r', alpha=0.5)
        self.axes.fill_between(x, -10, y, interpolate=True, color='g', alpha=0.5)
        plt.title(self.activation_function)
        # Display graph on plot
        self.canvas.draw()

    def slider_change_plot(self, weight1, weight2, bias):
        """Draw input points on graph"""
        # Axis setup
        self.axes.cla()
        self.axes.cla()
        self.axes.xaxis.set_visible(True)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        plt.xlabel("Input")
        plt.ylabel("Output")
        self.axes.scatter([self.input_array_x[0], self.input_array_x[1]],
                          [self.input_array_y[0], self.input_array_y[1]], s=150, marker='o', color='g',
                          edgecolor='none')
        self.axes.scatter([self.input_array_x[2], self.input_array_x[3]],
                          [self.input_array_y[2], self.input_array_y[3]], s=150, marker='o', color='r',
                          edgecolor='none')
        x = np.linspace(-10, 10)
        if weight2 == 0:
            y = -(bias + 0)
        else:
            y = -(bias + (weight1 * x) / weight2)
        #self.axes.plot(x, y)
        self.axes.fill_between(x, y, 10, interpolate=True, color='g', alpha=0.5)
        self.axes.fill_between(x, -10, y, interpolate=True, color='r', alpha=0.5)
        plt.title(self.activation_function)
        # Display graph on plot
        self.canvas.draw()

    def train_algorithm(self):
        """Trains by adjusting weights and bias upto fifty iterations"""
        ##################################################################
        # Credits:
        # Code logic: https://www.youtube.com/watch?v=1XkjVl-j8MM
        # Sample code: https://github.com/nsadawi/perceptron
        ##################################################################

        if self.activation_function == 'Symmetrical hard limit':
            # Clear plot before each new training.
            plt.cla()
            learn = 0
            for temp in range(100):
                error1 = 0
                for temp1 in range(4):
                    sigma_add = self.calculate_net_value(self.input_array_x[temp1], self.input_array_y[temp1],
                                                         self.bias_array[0])
                    # Check 2: Slider no change.
                    if self.activation_function == 'Symmetrical hard limit':
                        activation_output = self.symmetrical_hard_limit(sigma_add)
                        error = self.calculate_error(temp1, activation_output)
                        self.weight_array[0] += error * self.input_array_x[temp1]
                        self.weight_array[1] += error * self.input_array_y[temp1]
                        self.bias_array[0] += error

                        # Squared error calculation --> Corrects error.
                        error1 += (error*error)
                print("  error 1 ")
                print(error1)
                self.display_input_points()
                self.display_activation_function()

                if (error1 == 0) or temp > 100:
                    break

                learn += 1
                print("Learn done:" + str(learn))

            print("Decision boundary")
            print(self.weight_array[0])
            print(self.weight_array[1])
            print(self.bias_array[0])
            self.display_input_points()
            self.display_activation_function()

        elif self.activation_function == 'Linear':
            # Clear plot before each new training.
            plt.cla()
            learn = 0
            for temp in range(100):
                error1 = 0
                for temp1 in range(4):
                    sigma_add = self.calculate_net_value(self.input_array_x[temp1], self.input_array_y[temp1],
                                                         self.bias_array[0])
                    # Check 2: Slider no change.
                    if self.activation_function == 'Linear':
                        activation_output = self.linear(sigma_add)
                        error = float("{0:.2f}".format(self.calculate_error(temp1, activation_output)))
                        self.weight_array[0] += error * self.input_array_x[temp1]
                        self.weight_array[1] += error * self.input_array_y[temp1]
                        self.bias_array[0] += error

                        error1 = error1 + error*error
                print("  error 1 ")
                print(error1)

                if (error1 == 0) or temp > 100:
                    break

                learn += 1
                print("Learn done:" + str(learn))

            print("Decision boundary")
            print(self.weight_array[0])
            print(self.weight_array[1])
            print(self.bias_array[0])
            self.display_input_points()
            self.display_activation_function()

        elif self.activation_function == 'Hyperbolic Tangent':
            # Clear plot before each new training.
            plt.cla()
            learn = 0
            for temp in range(100):
                error1 = 0
                for temp1 in range(4):
                    sigma_add = self.calculate_net_value(self.input_array_x[temp1], self.input_array_y[temp1],
                                                         self.bias_array[0])
                    # Check 2: Slider no change.
                    if self.activation_function == 'Hyperbolic Tangent':
                        activation_output = self.hyperbolic_tangent(sigma_add)
                        error = round(self.calculate_error(temp1, activation_output))
                        self.weight_array[0] += round(error * self.input_array_x[temp1], 2)
                        self.weight_array[1] += round(error * self.input_array_y[temp1], 2)
                        self.bias_array[0] += round(error, 2)

                        error1 += (error*error)
                print("  error 1 ")
                print(error1)

                if (error1 == 0) or temp > 100:
                    break

                learn += 1
                print("Learn done:" + str(learn))

            print("Decision boundary")
            print(self.weight_array[0])
            print(self.weight_array[1])
            print(self.bias_array[0])
            self.display_input_points()
            self.display_activation_function()

        else:
            print("Not in selection")

    def calculate_net_value(self, x, y, z):
        """Calculates net value of neuron"""
        sigma_add = x * self.weight_array[0] + y * self.weight_array[1] + self.bias
        return round(sigma_add,2)

    def calculate_error(self, temp1, activation_output):
        """Calculates error by subtraction of epoch output with expected output"""
        error = self.output_array[temp1] - activation_output
        if self.activation_function == 'Linear' and error > 1000:
            error = 1000
        return round(error, 2)

    def symmetrical_hard_limit(self, sigma_add):
        """Calculate symmetrical tangent activation function value[-1 or 1]"""
        return 1 if sigma_add > self.theta else -1

    def linear(self, sigma_add):
        """Returns linear activation function value"""
        if sigma_add > 1000:
            sigma_add = 1000

        return round(sigma_add, 2)

    def hyperbolic_tangent(self, sigma_add):
        """Returns hyperbolic tangent activation function value"""
        return (m.exp(sigma_add)-m.exp(-sigma_add)) / (m.exp(sigma_add)+m.exp(-sigma_add))

    # Weight1 slider method
    def input_weight1_slider_callback(self):
        """Assign slider value to weight1 variable"""
        self.input_weight1 = self.input_weight1_slider.get()
        #self.display_activation_function()
        self.slider_change_plot(self.input_weight1, self.input_weight2, self.bias)

    # Weight2 slider method
    def input_weight2_slider_callback(self):
        """Assign slider value to weight2 variable"""
        self.input_weight2 = self.input_weight2_slider.get()
        #self.display_activation_function()
        self.slider_change_plot(self.input_weight1, self.input_weight2, self.bias)

    def bias_slider_callback(self):
        """Assign bias slider value to bias variable"""
        self.bias = self.bias_slider.get()
        #self.display_activation_function()
        self.slider_change_plot(self.input_weight1, self.input_weight2, self.bias)

    def activation_function_dropdown_callback(self):
        """Assign drop down selection to
        activation function variable to (Symmetrical hard limit, Hyperbolic Tangent, Linear)"""
        self.activation_function = self.activation_function_variable.get()