import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

def solve_Rz_equal_weights(I):
    """
    Solve for R(z) given a finite set I of real numbers.
    The equation is: 0 = sum_{i in I} prod_{j in I \ {i}} (R(z) + 1/z - j) - n*z * prod_{j in I} (R(z) + 1/z - j)
    Returns possible solutions for R(z).
    """
    # Define symbolic variables
    z = sp.Symbol('z')
    w = sp.Symbol('w')

    # Cardinality of I
    n = len(I)

    # Compute the full product: prod_{j in I} (w - j)
    full_product = 1
    for j in I:
        full_product *= (w - j)

    # Compute the sum of products: sum_{i in I} prod_{j in I \ {i}} (w - j)
    sum_products = 0
    for i in I:
        product = 1
        for j in I:
            if j != i:
                product *= (w - j)
        sum_products += product

    # Form the equation: sum_products - n*z*full_product = 0
    equation = sum_products - n * z * full_product

    # Solve for w
    solutions_w = sp.solve(equation, w)

    # Compute R(z) = w - 1/z for each solution
    solutions_Rz = [sol - 1 / z for sol in solutions_w]

    return solutions_Rz

def solve_Rz_custom_weights(I, weights):
    """
    Solve for R(z) given a finite set I of real numbers.
    The equation is: 0 = sum_{i in I} w(i) * prod_{j in I \ {i}} (R(z) + 1/z - j) - z * prod_{j in I} (R(z) + 1/z - j)
    Returns possible solutions for R(z).
    """
    # Define symbolic variables
    z = sp.Symbol('z')
    w = sp.Symbol('w')

    # Compute the full product: prod_{j in I} (w - j)
    full_product = 1
    for j in I:
        full_product *= (w - j)

    # Compute the sum of products: sum_{i in I} prod_{j in I \ {i}} (w - j)
    sum_products = 0
    for i in I:
        product = 1
        for j in I:
            if j != i:
                product *= (w - j)
        sum_products += weights[i] * product

    # Form the equation: sum_products - z*full_product = 0
    equation = sum_products - z * full_product

    # Solve for w
    solutions_w = sp.solve(equation, w)

    # Compute R(z) = w - 1/z for each solution
    solutions_Rz = [sol - 1 / z for sol in solutions_w]

    return solutions_Rz


def n_Rz(R_z, n):
    return n * R_z

def solve_Gz(Rz):
    """
    Solve for G(z) given a R(z).
    The equation is: R(G(z)) + 1/G(z) - z = 0
    Args:
        Rz_solutions: R(z) expression.
    Returns:
        Dictionary mapping each R(z) to its corresponding G(z) solutions.
    """
    z = sp.Symbol('z')
    w = sp.Symbol('w')  # w represents G(z)

    equation = Rz.subs(z, w) + 1 / w - z
    solutions = sp.solve(equation, w)
    Gz_solutions =  [sp.simplify(sol) for sol in solutions]

    return Gz_solutions

# I = {-1, 1}
# weights = {-1: 0.5, 1: 0.5}
# measure = [I, weights]
#
# Rz_solutions = solve_Rz_custom_weights(measure[0], measure[1])
#
# for n in range(1,5):
#     Gz_solutions = solve_Gz(n_Rz(Rz_solutions[0], 4))
#
#     # Print solutions
#     print(f"Set I: {I}")
#     print(f"Number of solutions: {len(Gz_solutions)}")
#     for idx, sol in enumerate(Gz_solutions, 1):
#         print(f"Solution {idx} for R(z):")
#         print(sol)
#         #sp.pprint(sol)


def compute_density(Gz, x_range=(-10, 10), epsilon=1e-6, points=2000):
    """
    Compute the density rho(x) using the Stieltjes inversion formula: rho(x) = (1/pi) * Im[G(x + i*epsilon)].
    Args:
        Gz: SymPy expression for G(z).
        x_range: Tuple of (x_min, x_max) for real part of z.
        epsilon: Small positive value for Im(z).
        points: Number of points for x grid.
    Returns:
        x_vals: Array of x values.
        rho: Array of density values rho(x).
    """
    # Convert G(z) to a NumPy-compatible function
    Gz_func = sp.lambdify(z, Gz, modules=['numpy'])

    # Create x grid
    x_vals = np.linspace(x_range[0], x_range[1], points)

    # Evaluate G(x + i*epsilon)
    z_vals = x_vals + 1j * epsilon
    G_vals = Gz_func(z_vals)

    # Compute density: rho(x) = (1/pi) * Im[G(x + i*epsilon)]
    rho = (1 / np.pi) * np.imag(G_vals)

    rho = abs(rho)

    # Handle NaNs and infinities
    rho = np.nan_to_num(rho, nan=0.0, posinf=1.0, neginf=1.0)

    return x_vals, rho


def plot_density(x_vals, rho, measure, save_name=None):
    """
    Create a density plot of rho(x) versus x.
    Args:
        x_vals: Array of x values.
        rho: Array of density values rho(x).
    """
    fig = plt.figure(figsize=(8, 6))
    rho /= sum(rho)
    plt.plot(x_vals, rho)
    plt.fill_between(x_vals, rho, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('Gęstość')
    plt.yticks([0])
    plt.title(f'Gęstość rozkładu {measure}')
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    fig.savefig(f"density_plots/plot{save_name}.png", dpi=300)
    plt.close()


a = 1
I = {0, a}

lmbd = 1
n = 100
w1 = 1 - (lmbd / n)
w2 = (lmbd / n)
w_norm = w1 + w2
w1 = w1 / w_norm
w2 = w2 / w_norm

weights = {0: w1, a: w2}
Rz_solutions = solve_Rz_custom_weights(I, weights)
selected_Rz = Rz_solutions[0]
Gz_solutions = solve_Gz(n_Rz(selected_Rz,n))
z = sp.Symbol('z')
x_vals, rho = compute_density(Gz_solutions[0], x_range=(-1, 10), epsilon=1e-16)
plot_density(x_vals, rho, r'$\left[(1 - \frac{1}{100})\delta_{0} + \frac{1}{100}\delta_{1}\right]^{\boxplus 100}$',
             save_name='poisson100step')
#
# z = sp.Symbol('z')
# r1 = solve_Rz_custom_weights({-2, 2}, {-2: 0.5, 2: 0.5})[0]
# n1 = 1
# r2 = solve_Rz_custom_weights({-1, 1}, {-1: 0.5, 1: 0.5})[0]
# n2 = 2
# r3 = solve_Rz_custom_weights({-2, 2}, {-2: 0.5, 2: 0.5})[0]
# n3 = 0
# r4 = solve_Rz_custom_weights({-2, 2}, {-2: 0.5, 2: 0.5})[0]
# n4 = 0
#
# Rz = n1 * r1 + n2 * r2 + n3 * r3 + n4 * r4
# Gz_solutions = solve_Gz(Rz)
# x_vals, rho = compute_density(Gz_solutions[0], x_range=(-5, 5), epsilon=1e-16)
# plot_density(x_vals,
#              rho,
#              r'$\left[\frac{1}{2}(\delta_{-2} + \delta_{2})\right] \boxplus \left[\frac{1}{2}(\delta_{-2} + \delta_{2})\right]^{\boxplus 2}$',
#              save_name='roznosci3')