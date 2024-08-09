"""
~~~~ Created by Ben Kroul, 2024 ~~~~
Defines useful utility functions and constants. Run printModule(util) after importing to see full list of imports.
- CVALS: object of physics constants
- printModule
- timeIt
- binarySearch
- linearInterpolate
- uFormat: PDG-style formatting of numbers with uncertainty, or just to significant figures
- RSquared, NRMSE
- FuncWLabels & FuncAdder: objects to fit composite functions for signal modelling. 
"""
import numpy as np
from glob import glob
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import matplotlib.lines as mlines
#import plotly.graph_objects as go

import sys, time, os

from scipy import special  # for voigt function
from scipy.optimize import curve_fit


# -- CONSTANTS -- #
# these directories are supposed to change by the user
DATADIR      = "/Users/benkroul/Documents/Physics/Data/"
SAVEDIR      = "/Users/benkroul/Documents/Physics/plots/"
SAVEEXT      = ".png"
FIGSIZE      = (10,6)
TICKSPERTICK = 5
FUNCTYPE     = type(sum)

class justADictionary():
    def __init__(self, my_name):
        self.name = my_name
        self.c    = 2.99792458 # 1e8   m/s speed of lgiht
        self.h    = 6.62607015 # 1e-34 J/s Plancks constant,
        self.kB   = 1.380649   # 1e-23 J/K Boltzmanns constant, 
        self.e    = 1.60217663 # 1e-19 C electron charge in coulombs
        self.a    = 6.02214076 # 1e23  /mol avogadros number
        self.Rinf = 10973731.56816  # /m rydberg constant
        self.G    = 0.0 # m^3/kg/s^2 Gravitational constant
        self.neutron_proton_mass_ratio = 1.00137842     # m_n / m_p
        self.proton_electron_mass_ratio = 1836.15267343 # m_p / m_e
        self.wien = 2.89777 # 1e-3  m*K  peak_lambda = wien_const / temp
    
    def __str__(self):
        return self.name

CVALS = justADictionary("Useful Physics constants, indexed in class for easy access")

# IBM's colorblind-friendly colors
#           |   Red  |   Blue  |  Orange |  Purple | Yellow  |   Green |   Teal  | Grey
hexcolors = ['DC267F', '648FFF', 'FE6100', '785EF0', 'FFB000', '009E73', '3DDBD9', '808080']
mpl.rcParams['axes.prop_cycle'] = cycler('color', [mpl.colors.to_rgba('#' + c) for c in hexcolors])

def savefig(title):
    plt.savefig(SAVEDIR + title + SAVEEXT, bbox_inches='tight')


# -- GENERAL FUNCTIONS -- #
def printModule(module):
    """print a module AFTER IMPORTING IT"""
    print("all imports:")
    numListedPerLine = 3; i = 0
    imported_stuff = dir(module)
    lst = [] # list of tuples of thing, type
    types = []
    for name in imported_stuff:
        if not name.startswith('__'):  # ignore the default namespace variables
            typ = str(type(eval(name))).split("'")[1]
            lst.append((name,typ))
            if typ not in types:
                types.append(typ)
    for typ in types:
        rowstart = '  '+typ+'(s): '
        i = 0; row = rowstart
        for id in lst:
            if id[1] != typ: continue
            i += 1
            row += id[0] +', '
            if not i % numListedPerLine:
                print(row[:-2])
                row = rowstart
        if len(row) > len(rowstart):
            print(row[:-2])
        i += numListedPerLine

def timeIt(func):
    """@ timeIt: Wrapper to print run time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.clock_gettime_ns(0)
        res = func(*args, **kwargs)
        end_time = time.clock_gettime_ns(0)
        diff = (end_time - start_time) * 10**(-9)
        print(func.__name__, 'ran in %.6fs' % diff)
        return res
    return wrapper

def binarySearch(X_val, X: list|tuple|np.ndarray, decreasing=False):
    """
    For sorted X, returns index i such that X[:i] < X_val, X[i:] >= X_val
     - if decreasing,returns i such that    X[:i] > X_val, X[i:] <= X_val
    """
    l = 0; r = len(X) - 1
    #print(f"searching for {X_val}, negative={negative}")
    m = (l + r) // 2
    while r > l:  # common binary search
        #print(f"{l}:{r} is {X[l:r+1]}, middle {X[m]}")
        if X[m] == X_val:  # repeat elements of X_val in array
            break
        if decreasing: # left is always larger than right
            if X[m] > X_val:
                l = m + 1
            else:
                r = m - 1
        else:        # right is always larger than left
            if X[m] < X_val:
                l = m + 1
            else:
                r = m - 1
        m = (l + r) // 2
    if r < l:
        return l
    if m + 1 < len(X):  # make sure we are always on right side of X_val
        if X[m] < X_val and not decreasing:
            return m + 1
        if X[m] > X_val and decreasing:
            return m + 1
    if X[m] == X_val:  # repeat elements of X_val in array
        if decreasing:
            while m > 0 and X[m - 1] == X_val:
                m -= 1
        elif not decreasing:
            while m + 1 < len(X) and X[m + 1] == X_val:
                m += 1
    return m

# linear interpolate 1D with sorted X
def linearInterpolate(x,X,Y):
    """example: 2D linear interpolate by adding interpolations from both
    - """
    i = binarySearch(x,X)
    if i == 0: i += 1  # lowest ting but we interpolate backwards
    m = (Y[i]-Y[i-1])/(X[i]-X[i-1])
    b = Y[i] - m*X[i]
    return m*x + b


# - ---- -STATS FUNCTIONS

def RSquared(y, model_y):
    """R^2 correlation coefficient of data"""
    n = len(y)
    # get mean
    SSR = SST = sm = 0
    for i in range(n):
        sm += y[i]
    mean_y = sm / n
    for i in range(n):
        SSR += (y[i] - model_y[i])**2
        SST += (y[i] - mean_y)**2
    return 1 - (SSR / SST)

def NRMSE(y, model_y, normalize=True):
    """Root mean squared error, can be normalized by range of data"""
    n = len(y)
    sm = 0; min_y = y[0]; max_y = y[0]
    for i in range(n):
        if y[i] < min_y: min_y = y[i]
        if y[i] > max_y: max_y = y[i]
        sm += (y[i] - model_y[i])**2
    y_range = max_y - min_y
    val = np.sqrt(sm/n)
    if normalize: 
        val = val / y_range
    return val

# ----- TEXT MANIPULATION

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# method to return the string form of an integer (0th, 1st, 2nd, 3rd, 4th...)
Ith = lambda i: str(i) + ("th" if (abs(i) % 100 in (11,12,13)) else ["th","st","nd","rd","th","th","th","th","th","th"][abs(i) % 10])

def arrFromString(data, col_separator = '\t', row_separator = '\n'):
    """ Return numpy array from string
    - great for pasting Notion tables into np array """
    x = []; L = 0
    for row in data.split(row_separator):
        if len(row):  # ignore any empty rows
            entries = row.split(col_separator)
            newL = len(entries)
            if L != 0 and newL != L:
                print(f"Rows have different lengths {L} and {newL}:")
                print(x)
                print(entries)
            L = newL
            x.extend(entries)
    return np.reshape(np.array(x,dtype='float64'),(-1,L))

def uFormat(number, uncertainty=0, figs = 4, shift = 0, FormatDecimals = False):
    """
    Returns "num_rounded(with_sgnfcnt_dgts_ofuncrtnty)", formatted to 10^shift
      or number rounded to figs significant figures if uncertainty = 0
    According to section 5.3 of "https://pdg.lbl.gov/2011/reviews/rpp2011-rev-rpp-intro.pdf"

    Arguments:
    - float number:      the value
    - float uncertainty: the absolute uncertainty (stddev) in the value
       - if zero, will just format number to figs significant figures (see figs)
    - int figs: when uncertainty = 0, format number to degree of sig figs instead
       - if zero, will simply return number as string
    - int shift:  optionally, shift the resultant number to a higher/lower digit expression
       - i.e. if number is in Hz and you want a string in GHz, specify shift = 9
               likewise for going from MHz to Hz, specify shift = -6
    - bool FormatDecimals:  for a number 0.00X < 1e-2, option to express in "X.XXe-D" format
             for conciseness. doesnt work in math mode because '-' is taken as minus sign
    """
    num = str(number); err = str(uncertainty)
    
    sigFigsMode = not uncertainty    # UNCERTAINTY ZERO: IN SIG FIGS MODE
    if sigFigsMode and not figs: # nothing to format
        return num
    
    negative = False  # add back negative later
    if num[0] == '-':
        num = num[1:]
        negative = True
    if err[0] == '-':  # stddev is symmetric ab number
        err = err[1:]
    
    # ni = NUM DIGITS to the RIGHT of DECIMAL
    # 0.00001234=1.234e-4 has ni = 8, 4 digs after decimal and 4 sig figs
    # 1234 w/ ni=5 corresponds to 0.01234
    ni = ei = 0  
    if 'e' in num:
        ff = num.split('e')
        num = ff[0]
        ni = -int(ff[1])
    if 'e' in err:
        ff = err.split('e')
        err = ff[0]
        ei = -int(ff[1])

    if not num[0].isdigit():
        print(f"uFormat: {num} isn't a number")
        return num
    if not err[0].isdigit():
        err = '?'

    # comb through error, get three most significant figs
    foundSig = False; decimal = False
    topThree = ""; numFound = 0
    jErr = ""
    for ch in err:
        if decimal:
            ei += 1
        if not foundSig and ch == '0': # dont care ab leading zeroes
            continue  
        if ch == '.':
            decimal = True
            continue
        jErr += ch
        if numFound >= 3:  # get place only to three sigfigs
            ei -= 1
            continue
        foundSig = True
        topThree += ch
        numFound += 1
    
    foundSig = False; decimal = False
    jNum = ""
    for ch in num:
        if decimal:
            ni += 1
        if not foundSig and ch == '0': # dont care ab leading zeroes
            continue
        if ch == '.':
            decimal = True
            continue
        jNum += ch
        foundSig = True
    if len(jNum) == 0:  # our number is literally zero!
        return '0'
    
    # round error correctly according to PDG
    if len(topThree) == 3:
        nTop = int(topThree)
        if nTop < 355: # 123 -> (12.)
            Err = int(topThree[:2])
            if int(topThree[2]) >= 5:
                Err += 1
            ei -= 1
        elif nTop > 949: # 950 -> (10..)
            Err = 10
            ei -= 2
        else:  # 355 -> (4..)
            Err = int(topThree[0])
            if int(topThree[1]) >= 5:
                Err += 1
            ei -= 2
        Err = str(Err)
    else:
        Err = topThree

    n = len(jNum); m = len(Err)
    nBefore = ni - n  #; print(num, jNum, n, ni, nBefore)
    eBefore = ei - m  #; print(err, Err, m, ei, eBefore)
    if nBefore > eBefore:  # uncertainty is a magnitude larger than number, still format number
        if not sigFigsMode:
            print(f'Uncrtnty: {uncertainty} IS MAGNITUDE(S) > THAN Numba: {number}')
        Err = '?'
    if sigFigsMode or nBefore > eBefore:
        ei = nBefore + figs

    # round number to error
    d = ni - ei 
    if ni == ei: 
        Num = jNum[:n-d]
    elif d > 0:  # error has smaller digits than number = round number
        Num = int(jNum[:n-d])
        if int(jNum[n-d]) >= 5:
            Num += 1
        Num = str(Num)
    else:  # error << num
        Num = jNum
        if ei < m + ni:
            Err = Err[n+d-1]
        else:
            Err = '0'
    if ni >= ei: ni = ei  # indicate number has been rounded

    n = len(Num)
    # if were at <= e-3 == 0.009, save formatting space by removing decimal zeroes
    extraDigs = 0
    if not shift and FormatDecimals and (ni-n) >= 2:
        shift -= ni - n + 1
        extraDigs = ni - n + 1
    
    # shift digits up/down by round argument
    ni += shift
    end = ''

    # there are digits to the right of decimal and we dont 
    # care about exact sig figs (ex. cut zeroes from 0.02000)
    if ni > 0 and not sigFigsMode:
        while Num[-1] == '0':
            if len(Num) == 1: break
            Num = Num[:-1]
            ni -= 1
            n -= 1
    
    if ni >= n:   # place decimal before any digits
        Num = '0.' + "0"*(ni-n) + Num
    elif ni > 0:  # place decimal in-between digits
        Num = Num[:n-ni] + '.' + Num[n-ni:]
    elif ni < 0:  # add non-significant zeroes after number
        end = 'e'+str(-ni)
    if extraDigs:  # format removed decimal zeroes
        end = 'e'+str(-extraDigs)
    
    if negative: Num = '-' + Num  # add back negative
    if not sigFigsMode:
        end = '(' + Err + ')' + end
    return Num + end

#print("da master physics/CS folder - good luck code monkey")
if __name__ == "__main__":
    # behold my pride and joy - uFormat
    while True:
        t = input("Enter space-separated arguments to uFormat in the order of \n\
                  number, uncertainty, sig_figs=4, shift=0, FormatDecimals=T/F)\n:")
        if not t:
            break 
        args = t.rstrip().split(' ')
        i = 0; cs = []
        while i < min(len(args),6):
            if i < 2:
                cs.append(float(args[i]))
            elif i < 4:
                cs.append(int(args[i]))
            else:
                cs.append('t' in args[i].lower())
            i += 1
        print(uFormat(*cs))
            