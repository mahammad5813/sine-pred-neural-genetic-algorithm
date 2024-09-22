import numpy as np
import matplotlib.pyplot as plt
from activations import tanh

def plot_func(func, min=-10, max=10, step=0.1):
    x = np.arange(min,max,step)
    y = [func([i]) for i in x]
    plt.plot(x,y)
    plt.show()

def plot_the_best(nn, min=-10, max=10, step=0.1):
    x = np.arange(min,max,step)
    y = [nn.evaluate(i)[0] for i in x]
 
    plt.plot(x,y)
    plt.show()

def plot_bunch_nn(bunch, min=-10, max=10, step=0.1, n_cols=10):
    x = np.arange(min, max, step)
    fig, ax = plt.subplots(int(len(bunch)/n_cols)+1, n_cols)
    n=0
    run = True
    for row in ax:
        if not run:
            break
        for col in row:
            y = [bunch[n].evaluate(i)[0] for i in x]

            col.plot(x, y)
            n+=1
            if n >= len(bunch):
                run=False
                break
    plt.show()

if __name__ == '__main__':
    x = np.arange(-10,10,0.1)
    y = [tanh(i) for i in x]
    plt.plot(x,y)
    plt.show()
