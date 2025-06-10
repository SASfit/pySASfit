import tkinter as tk
from tkinter import filedialog
from SASformats import read_Ascii, create_ASCIIData
import easygui
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
#file_path = easygui.fileopenbox()
#print(file_path)

"""
def select_and_read_ascii():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select ASCII Data File",
        filetypes=[("ASCII files", "*.dat *.txt *.asc"), ("All files", "*.*")]
    )
    if not file_path:
        print("No file selected.")
        return None, None

    data, AsciiCont = read_Ascii(file_path)
    if data == 0:
        print("File could not be read or format is invalid.")
        return None, None
    print(f"File '{file_path}' loaded successfully.")
    return data, AsciiCont
"""
def select_and_read_ascii():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select ASCII Data File",
        "",
        "ASCII files (*.dat *.asc);;All files (*)"
    )
    if not file_path:
        print("No file selected.")
        return None, None

    data, AsciiCont = read_Ascii(file_path)
    if data == 0:
        print("File could not be read or format is invalid.")
        return None, None
    print(f"File '{file_path}' loaded successfully.")
    return data, AsciiCont

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d
import os

errf, ascii_data = select_and_read_ascii()
if ascii_data is not None:
    plt.plot(ascii_data.x, ascii_data.y, marker='o', linestyle='-', markersize=1)
    plt.xlabel("r")
    plt.ylabel("eta")
    plt.title("ASCII Data Plot")
    plt.show()
    
    Q = np.logspace(-3, 1, num=1000)  # from 10^-3 to 10^1
    r = np.array(ascii_data.x)
    eta = np.array(ascii_data.y)

        # For each Q, integrate over r: Int[eta(r) * 4*pi*r^2 * sinc(Q*r)] dr
    F_Q = []
    for q in Q:
        integrand = eta * 4 * np.pi * r**2 * np.sinc(q * r / np.pi)
        # I_Q.append(np.trapezoid(integrand, r))
        F_Q.append(scipy.integrate.simpson(integrand, r))
        #f_interp = interp1d(r, integrand, kind='cubic', fill_value=0, bounds_error=False)
        #result, _ = scipy.integrate.quad(f_interp, r.min(), r.max(), limit=200)
        #I_Q.append(result)
    F_Q = np.array(F_Q)
    I_Q = 1.90721e-08 *np.abs(F_Q)**2
    plt.loglog(Q, I_Q, marker='o', linestyle='-', markersize=1)
    plt.xlabel("Q")
    plt.ylabel("I(Q)")
    plt.title("ASCII Data Plot")
    plt.show()
    base, ext = os.path.splitext(ascii_data.FileName)
    output_path = base + '_IQ.'+ext
    np.savetxt(output_path, np.column_stack((Q, I_Q)), header='Q I(Q)')