from data_preprocessing import DataProcessor
from k_means import KMeans
import numpy as np

from gui import GUI
import tkinter as tk

def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
