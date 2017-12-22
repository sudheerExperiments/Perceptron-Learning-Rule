# Hanuma Venkata Sai Sudheer, Raja
# 1001-541-257
# 2017-09-17
# Assignment_01_01

import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import Raja_01_02 as r02


class WidgetsWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        # Status bar
        self.status_bar = StatusBar(self, self, bg='yellow', bd=1, relief=tk.SUNKEN)

        self.center_frame = tk.Frame(self)
        # Create a frame for plotting graphs
        self.left_frame = PlotsDisplayFrame(self, self.center_frame, bg='blue')
        self.display_activation_functions = r02.DisplayActivationFunctions(self, self.left_frame)
        # Create a frame for displaying graphics
        #self.right_frame = GraphicsDisplayFrame(self, self.center_frame, bg='yellow')
        self.center_frame.grid(row=1, column=0,sticky=tk.N + tk.E + tk.S + tk.W)
        self.center_frame.grid_propagate(True)
        self.center_frame.rowconfigure(1, weight=1, uniform='xx')
        #self.center_frame.columnconfigure(0, weight=1, uniform='xx')
        self.center_frame.columnconfigure(1, weight=1, uniform='xx')
        self.status_bar.grid(row=2, column=0,sticky=tk.N + tk.E + tk.S + tk.W)
        self.status_bar.rowconfigure(2, minsize=20)
        self.left_frame.grid(row=0, column=0,sticky=tk.N + tk.E + tk.S + tk.W)
        #self.right_frame.grid(row=0, column=1,sticky=tk.N + tk.E + tk.S + tk.W)


class StatusBar(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.label = tk.Label(self)
        self.label.grid(row=0,sticky=tk.N + tk.E + tk.S + tk.W)

    def set(self, format, *args):
        self.label.config(text=format % args)
        self.label.update_idletasks()

    def clear(self):
        self.label.config(text="")
        self.label.update_idletasks()


class PlotsDisplayFrame(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        # Removed right display area(yellow frame)


def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

widgets_window = WidgetsWindow()
#widgets_window.geometry("500x500")
# widgets_window.wm_state('zoomed')
widgets_window.title('Perceptron learn rule')
widgets_window.minsize(600,300)
widgets_window.protocol("WM_DELETE_WINDOW", lambda root_window=widgets_window: close_window_callback(root_window))
widgets_window.mainloop()
