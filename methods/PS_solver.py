import numpy as np

def proj_unit_simplex_median(y):
    """
    Project y onto the simplex:
        {x : x >= 0, sum(x) = z}

    For unit simplex, use z = 1.

    Parameters
    ----------
    y : array-like, shape (n,)
        Input vector.
    z : float
        Radius of simplex. For unit simplex, z = 1.
    verbose : bool
        If True, print the screening process.

    Returns
    -------
    x : ndarray
        Projection of y onto the simplex.
    theta : float
        Threshold value.
    """

    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional vector.")

    z = 1.0

    verbose = False

    # U stores candidates whose active/inactive status is not determined yet
    U = y.copy()

    # s is the sum of elements that have been confirmed active
    # rho is the number of elements confirmed active
    s = 0.0
    rho = 0

    while U.size > 0:
        # Choose the median as pivot
        pivot = np.median(U)

        # Split U into two groups
        G = U[U >= pivot]   # larger group
        L = U[U < pivot]    # smaller group

        sum_G = np.sum(G)
        rho_G = len(G)

        # Suppose all elements in G are active.
        # Then the corresponding threshold is:
        theta_candidate = (s + sum_G - z) / (rho + rho_G)

        if verbose:
            print("pivot =", pivot)
            print("theta_candidate =", theta_candidate)
            print("rho =", rho, "rho_G =", rho_G)
            print("s =", s, "sum_G =", sum_G)
            print()

        if theta_candidate < pivot:
            # Then all elements in G are indeed active.
            # Add them into the active set.
            s += sum_G
            rho += rho_G

            # Continue checking smaller elements.
            U = L
        else:
            # Then elements <= pivot cannot be active.
            # Only elements strictly larger than pivot may still be active.
            U = U[U > pivot]

    # Final threshold
    theta = (s - z) / rho

    # Projection
    x = np.maximum(y - theta, 0)

    return x




def w(JF, fy, Fx, yk, L, l,  prox, x):   #Frank-Wolfe算法中的目标函数  g = 1/n*\|\cdot\|_{1}
    A = np.dot(JF, JF.T)
    s = prox - yk + np.dot(JF.T, x) / L
    Moreau = 0.5 * np.dot(s,s)
    return 0.5 / L * np.dot(x, np.dot(A, x)) + np.dot(x, Fx - fy) - L * Moreau

def cond_gra_simp_prox(JF, fy, Fx, yk, L, l, lam):  #proximal的away-step条件梯度法,  l为正则项系数，只考虑LASSO问题
    # print or not
    P = False
    # P = True
    Bull = True
    x = lam
    ite = 0
    ite_max = 30 #设置内循环最大次数
    n = len(x)
    x_old = np.random.dirichlet(np.ones(n))
    prox_old = proj_unit_simplex_median(yk - np.dot(JF.T, x_old) / L) # prox_{\lambda g}(x_k - \sum\lambda\nabla f(x))
    grad_old = Fx - fy - np.dot(JF, prox_old - yk)
    # print("a")
    while Bull:
        # print("b")
        Bull = False
        prox = proj_unit_simplex_median(yk - np.dot(JF.T, x) / L)  # prox_{\lambda g}(x_k - \sum\lambda\nabla f(x))
        grad = Fx - fy - np.dot(JF, prox - yk)

        alpha = np.dot(grad - grad_old, x - x_old) / np.dot(x - x_old, x - x_old)
        alpha = np.maximum(alpha,1e-5)
        # if P:
        #     print(alpha, x, x_old)

        grad_old = grad
        x_old = x
        # if P:
            # print("aaa",grad,x)

        d = proj_unit_simplex_median(x - 1 / alpha * grad) - x
        # if P:
            # print("bbb")

        if alpha * np.linalg.norm(d) > 1e-9 and np.linalg.norm(d) > 1e-12:
            Bull = True
            t = 1
            Bull2 = True
            iite = 0
            while Bull2:
                # print("c")
                iite += 1
                Bull2 = False
                prox_new = proj_unit_simplex_median(yk - np.dot(JF.T, x + t * d) / L)
                if w(JF, fy, Fx, yk, L, l, prox_new, x + t * d) - w(JF, fy, Fx, yk, L, l, prox, x) > 1e-4 * t * np.dot(
                        grad, d):
                    Bull2 = True
                    t = 0.5 * t
                if iite >= 5:
                    Bull2 = False
                if P:
                    print(iite)
            x = x + t * d
            ite += 1

        if P:
            print("ite1", ite,np.linalg.norm(d),alpha)


        if ite >= ite_max:
            Bull = False
    return x
#######################################################################################################################


def w1(JF, g, xk, l,  prox, x):   #Frank-Wolfe算法中的目标函数  g = 1/n*\|\cdot\|_{1}
    A = np.dot(JF, JF.T)
    s = prox - xk + np.dot(JF.T, x)
    Moreau = 0.5 * np.dot(s,s)
    return 0.5 * np.dot(x, np.dot(A, x)) + np.dot(x, g) - Moreau

def cond_gra_simp_prox1(JF, g, xk, l, lam):  #proximal的away-step条件梯度法,  l为正则项系数，只考虑LASSO问题
# print or not
    P = False
    # P = True
    Bull = True
    x = lam
    ite = 0
    ite_max = 30  # 设置内循环最大次数
    n = len(x)
    x_old = np.random.dirichlet(np.ones(n))
    prox_old = proj_unit_simplex_median(xk - np.dot(JF.T, x_old))  # prox_{\lambda g}(x_k - \sum\lambda\nabla f(x))
    grad_old = g - np.dot(JF, prox_old - xk)
    while Bull:
        Bull = False
        prox = proj_unit_simplex_median(xk - np.dot(JF.T, x))  # prox_{\lambda g}(x_k - \sum\lambda\nabla f(x))
        grad = g - np.dot(JF, prox - xk)

        alpha = np.dot(grad - grad_old, x - x_old) / np.dot(x - x_old, x - x_old)
        alpha = np.maximum(alpha, 1e-5)
        # print(alpha, x, x_old)

        grad_old = grad
        x_old = x

        d = proj_unit_simplex_median(x - 1 / alpha * grad) - x
        if alpha * np.linalg.norm(d) > 1e-9 and np.linalg.norm(d) > 1e-12:
            Bull = True
            t = 1
            Bull2 = True
            iite = 0
            while Bull2:
                iite += 1
                Bull2 = False
                prox_new = proj_unit_simplex_median(xk - np.dot(JF.T, x + t * d))
                if w1(JF, g, xk, l, prox_new, x + t * d) - w1(JF, g, xk, l, prox, x) > 1e-4 * t * np.dot(grad, d):
                    Bull2 = True
                    t = 0.5 * t
                if iite >= 10:
                    Bull2 = False
            x = x + t * d
            ite += 1
        if P:
            print("ite2", ite, np.linalg.norm(d), alpha)
        if ite >= ite_max:
            Bull = False
    return x
#######################################################################################################################