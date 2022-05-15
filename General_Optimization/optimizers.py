'''
Module implementing several optimization methods
'''

from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

def dichotomic_search(
    fobj: Callable[[float], float],
    interval: Tuple[float,float],
    eps: float,
    l_toler: float,
    plot=False,
) -> Tuple[float,float]:

    """
    Solve a 1D optimization problem using the dichotomic search method

    Parameters
    ----------
    fobj:
        Objective function to be optimized
    interval:
        Region to consider during optimization
    eps:
        Half the distance between lambda and mu points (see algorithm)
        ATTENTION: must be smaller than l_toler/2 or algorithm wouldn't converge
    l_toler:
        Length of interval under with the algorithm halts
    plot:
        Boolean flag that controls graphic display of the algorithm

    Returns
    -------
    final_interval:
        Interval of <= desired length containing the solution

    """
    assert eps<l_toler/2, \
          "eps must be smaller than l_toler/2 (see docstring)"

    a,b = interval
    if plot:
        x_ax = np.linspace(a,b,1000)
        y_ax = fobj(x_ax)
        fig,ax = plt.subplots(1,1)
        ax.plot(x_ax,y_ax,label="Objective function")
        ax.set_ylabel("$f_{obj}(x)$")
        ax.set_xlabel("$x$")
        ax.set_title("Dichotomic Search")

    while b - a > l_toler:
        lamb = (a+b)/2 - eps
        mu = lamb + 2*eps
        if fobj(lamb) < fobj(mu):
            b = mu
        else:
            a = lamb
        if plot:
            height = max(fobj(a),fobj(b))
            ax.plot([a,b],[height,height], "|-",color="black", lw=0.75)
            ax.legend()
    return a,b


def aureus_section(
    fobj: Callable[[float], float],
    interval: Tuple[float,float],
    l_toler: float,
    plot=False,
) -> Tuple[float,float]:

    """
    Solve a 1D optimization problem using the aureus section method

    Parameters
    ----------
    fobj:
        Objective function to be optimized
    interval:
        Region to consider during optimization
    l_toler:
        Length of interval under with the algorithm halts
    plot:
        Boolean flag that controls graphic display of the algorithm

    Returns
    -------
    final_interval:
        Interval of <= desired length containing the solution

    """
    alpha = 0.618
    a,b = interval

    if plot:
        x_ax = np.linspace(a,b,1000)
        y_ax = fobj(x_ax)
        fig,ax = plt.subplots(1,1)
        ax.plot(x_ax,y_ax,label="Objective function")
        ax.set_ylabel("$f_{obj}(x)$")
        ax.set_xlabel("$x$")
        ax.set_title("Aureus Section")
        ax.legend()

    while b - a > l_toler:
        lamb = a + (1-alpha)*(b-a)
        mu = a + alpha*(b-a)
        if fobj(lamb) < fobj(mu):
            b = mu
        else:
            a = lamb
        if plot:
            height = max(fobj(a),fobj(b))
            ax.plot([a,b],[height,height], "|-",color="black", lw=0.75)
    return a,b


def hooke_jeeves(
    fobj: Callable[[np.ndarray],float],
    eps: float,
    interval: Tuple[np.ndarray,np.ndarray],
    init: np.ndarray,
    linear_solver = dichotomic_search,
    plot=False,
    plot_fobj=False,
) -> np.ndarray:

    """
    Solve a ND optimization problem using the Hooke-Jeeves algorithm

    Parameters
    ----------
    fobj:
        Objective function to be optimized
    eps:
        Minimum difference between each solution point required to keep going
    interval:
        Region to be considered during optimization
    init:
        Starting point for optimization
    linear_solver:
        Algorithm used to solve the linear optimization problem
    plot:
        Boolean flag that controls graphic display of the algorithm

    Returns
    -------
    solution:
        Optimum solution found

    """
    assert eps>0, \
           "eps must be a positive constant (see docstring)"

    lw_bounds, up_bounds = interval
    dim = len(init)

    if plot:
        assert len(init) == 2, \
            "Objective function must be two dimentional"

        x_ax = np.linspace(lw_bounds[0],up_bounds[0],1000)
        y_ax = np.linspace(lw_bounds[1],up_bounds[1],1000)
        fig,ax = plt.subplots(1,1)
        if plot_fobj:
            X,Y = np.meshgrid(x_ax,y_ax)
            Z = plot_fobj(X,Y)
            ax.contour(X,Y,Z)

        ax.plot(init[0],init[1], "o", color="blue", label="Starting point",
        markersize=4)
        ax.set_ylabel("$y$")
        ax.set_xlabel("$x$")
        ax.set_title("Hooke - Jeeves")

    xi = np.inf
    xf = init
    y = init
    while np.sum(np.abs(xf - xi)) >= eps:
        for j in range(dim):
            d = np.zeros(dim)
            d[j] = 1
            step_fobj = lambda lamb: fobj(y + d*lamb)
            lamb = linear_solver(step_fobj,
                                (lw_bounds[j],up_bounds[j]),
                                eps/2.5, eps)
            lamb = np.mean(lamb)
            if plot:
                ax.arrow(y[0],y[1],lamb*d[0],lamb*d[1], width=0.02, head_width=0.12,
                length_includes_head=True, fc='black')
            y = y + lamb*d

        xi = xf
        xf = y
        if np.sum(np.abs(xf - xi)) < eps:
            break
        d = xf - xi
        step_fobj = lambda lamb: fobj(xf + d*lamb)

        lw_d_bounds, up_d_bounds = [], []
        with np.errstate(divide='ignore'):
            for j in range(dim):
                lw_d_bounds.append(lw_bounds[j]/d[j] - xf[j])
                up_d_bounds.append(up_bounds[j]/d[j] - xf[j])
        lwb, upb = min(lw_d_bounds), max(up_d_bounds)
        lamb = linear_solver(step_fobj,
                            (lwb,upb),
                            eps/2.5, eps)
        lamb = np.mean(lamb)
        y = xf + lamb*d
        if plot:
            ax.arrow(xf[0],xf[1],lamb*d[0],lamb*d[1], width=0.02, head_width=0.12,
            length_includes_head=True, fc='black')
    if plot:
        ax.plot(xf[0],xf[1], "*", color="darkorange", label="Optimal solution found")
        ax.legend()
    solution = xf
    return solution


def gradient(
    fobj: Callable[[np.ndarray],float],
    grad_fobj: Callable[[np.ndarray],float],
    eps: float,
    interval: Tuple[np.ndarray,np.ndarray],
    init: np.ndarray,
    linear_solver = dichotomic_search,
    plot=False,
    plot_fobj=False
) -> np.ndarray:

    """
    Solve a ND optimization problem using the gradient algorithm

    Parameters
    ----------
    fobj:
        Objective function to be optimized
    grad_fobj:
        Gradient function of the objective function to be optimized
    eps:
        Minimum difference between each solution point required to keep going
    interval:
        Region to be considered during optimization
    init:
        Starting point for optimization
    linear_solver:
        Algorithm used to solve the linear optimization problem
    plot:
        Boolean flag that controls graphic display of the algorithm

    Returns
    -------
    solution:
        Optimum solution found

    """

    assert eps>0, \
           "eps must be a positive constant (see docstring)"

    li,ls = interval

    if plot:
        assert len(init) == 2, \
            "Objective function must be two dimentional"
        x_ax = np.linspace(li[0],ls[0],1000)
        y_ax = np.linspace(li[1],ls[1],1000)
        X,Y = np.meshgrid(x_ax,y_ax)
        Z = plot_fobj(X,Y)
        fig,ax = plt.subplots(1,1)
        ax.contour(X,Y,Z)#,label="Objective function")
        ax.plot(init[0],init[1], "o", color="blue", label="Starting point",
        markersize=4)
        ax.set_ylabel("$y$")
        ax.set_xlabel("$x$")
        ax.set_title("Gradient")

    li = min(li)
    ls = max(ls)
    xi = init

    while np.linalg.norm(grad_fobj(xi)) > eps:
        di = -grad_fobj(xi)

        step_fobj = lambda lamb: fobj(xi + di*lamb)
        lambdai = linear_solver(step_fobj,
                            (li,ls),
                            eps/2.5, eps)
        lambdai = np.mean(lambdai)

        xf = xi + lambdai*di
        x_dif = xf-xi

        if plot:
          ax.arrow(xi[0],xi[1],x_dif[0],x_dif[1], width=0.02, head_width=0.12,
                    length_includes_head=True, fc='black')
        xi = xf

    if plot:
        ax.plot(xi[0],xi[1], "*", color="darkorange", label="Optimal solution found")
        ax.legend()

    solution = xi
    return solution


def DFP(
    fobj: Callable[[np.ndarray],float],
    grad_fobj: Callable[[np.ndarray],float],
    eps: float,
    interval: Tuple[np.ndarray,np.ndarray],
    init: np.ndarray,
    D_init: np.matrix,
    linear_solver = dichotomic_search,
    plot=False,
    plot_fobj=False
) -> np.ndarray:

    """
    Solve a ND optimization problem using the gradient algorithm

    Parameters
    ----------
    fobj:
        Objective function to be optimized
    grad_fobj:
        Gradient function of the objective function to be optimized
    eps:
        Minimum difference between each solution point required to keep going
    interval:
        Region to be considered during optimization
    init:
        Starting point for optimization
    D_init:
        Starting D simetrical matrix
    linear_solver:
        Algorithm used to solve the linear optimization problem
    plot:
        Boolean flag that controls graphic display of the algorithm

    Returns
    -------
    solution:
        Optimum solution found

    """

    assert eps>0, \
           "eps must be a positive constant (see docstring)"

    li,ls = interval

    if plot:
        assert len(init) == 2, \
            "Objective function must be two dimentional"
        x_ax = np.linspace(li[0],ls[0],1000)
        y_ax = np.linspace(li[1],ls[1],1000)
        X,Y = np.meshgrid(x_ax,y_ax)
        Z = plot_fobj(X,Y)
        fig,ax = plt.subplots(1,1)
        ax.contour(X,Y,Z)#,label="Objective function")
        ax.plot(init[0],init[1], "o", color="blue", label="Starting point",
        markersize=4)
        ax.set_ylabel("$y$")
        ax.set_xlabel("$x$")
        ax.set_title("Davidon-Fletcher-Powell")

    li = min(li)
    ls = max(ls)
    xi = init
    xf = np.zeros(np.shape(xi))
    Di = D_init

    while np.linalg.norm(grad_fobj(xi)) > eps:
        di = -Di @ grad_fobj(xi)

        for j,dj in enumerate(di):
            x_ = np.zeros(np.shape(xi))
            x_[j] = 1
            step_fobj = lambda lamb: fobj(xi + dj*x_*lamb)
            lambdai = linear_solver(step_fobj,
                                (li,ls),
                                eps/2.5, eps)
            lambdai = np.mean(lambdai)
            xf[j] = xi[j] + lambdai*dj

        p = (xf-xi)[None,:]
        q = (grad_fobj(xf)-grad_fobj(xi))[None,:]
        Di = Di + (p.T @ p)/np.dot(p.flatten(),q.flatten()) - (Di @ q.T @ q @ Di)/np.dot(q.flatten(), (q @ Di.T).flatten())

        if plot:
          ax.arrow(xi[0],xi[1],p[0][0],p[0][1], width=0.02, head_width=0.12,
                    length_includes_head=True, fc='black')
        xi = xf

    if plot:
        ax.plot(xi[0],xi[1], "*", color="darkorange", label="Optimal solution found")
        ax.legend()

    solution = xi
    return solution
