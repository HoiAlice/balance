import jax
import jax.numpy as jnp
from jax import lax, jit
from functools import partial
from dataclasses import dataclass
import matplotlib.pyplot as plt

# =============================================================
# System parameters as pytree for JAX
# =============================================================
@dataclass
class SystemParameters:
    alpha: float
    gamma: float
    delta: float
    E: float

# Register as pytree so JAX can handle it
@jax.tree_util.register_pytree_node_class
class JaxSystemParameters(SystemParameters):
    def tree_flatten(self):
        children = (self.alpha, self.gamma, self.delta, self.E)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# =============================================================
# Aggregation operator F(k) for fixed point
# =============================================================
@jit
def F(k_next: float, s_prev: jnp.ndarray, params: JaxSystemParameters) -> float:
    alpha = params.alpha
    gamma = params.gamma
    delta = params.delta
    E = params.E

    N = s_prev.size
    e = E / N
    xi = (1.0 - alpha) / alpha

    k_t = jnp.mean(s_prev)
    R_t = alpha * k_t ** (alpha - 1.0)
    w_t = (1 - alpha) * k_t ** alpha

    R_next = alpha * k_next ** (alpha - 1.0)
    w_next = (1 - alpha) * k_next ** alpha

    inside = R_t * s_prev + w_t + e - gamma * ((xi + 1.0) * R_t * k_t - k_next + e) - (w_next + e) / (delta * R_next)
    return jnp.mean(jnp.maximum(inside, 0.0)) * delta / (1 + delta)


# =============================================================
# Fixed point solver: binary search
# =============================================================
@partial(jit, static_argnames=("max_iter", "tol"))
def solve_k_bisect(s_prev: jnp.ndarray, params: JaxSystemParameters, max_iter: int = 1000, tol: float = 1e-8):
    N = s_prev.size
    e = params.E / N

    k_lo = 0.0
    k_hi = (1.0 - params.gamma) * (jnp.mean(s_prev) ** params.alpha) + e

    def cond_fn(val):
        k_lo, k_hi, i = val
        return (i < max_iter) & ((k_hi - k_lo) > tol)

    def body_fn(val):
        k_lo, k_hi, i = val
        k_mid = 0.5 * (k_lo + k_hi)
        f_mid = F(k_mid, s_prev, params)
        k_lo_new = jnp.where(f_mid > k_mid, k_mid, k_lo)
        k_hi_new = jnp.where(f_mid > k_mid, k_hi, k_mid)
        return k_lo_new, k_hi_new, i + 1

    k_lo, k_hi, _ = lax.while_loop(cond_fn, body_fn, (k_lo, k_hi, 0))
    return 0.5 * (k_lo + k_hi)


# =============================================================
# Fixed point solver: simple iteration (Picard)
# =============================================================
@partial(jit, static_argnames=("max_iter", "tol"))
def solve_k_iterative(
    s_prev: jnp.ndarray,
    params: JaxSystemParameters,
    max_iter: int = 100000,
    tol: float = 1e-8,
    damping: float = 1.0,   # relaxation parameter in (0,1]
):
    """
    Solve k = F(k) using damped fixed-point iteration:
        k_{n+1} = (1 - damping) * k_n + damping * F(k_n)
    """

    # Initial guess (same upper bound you used for bisection)
    k0 = jnp.mean(s_prev)

    def cond_fn(val):
        k_old, k_new, i = val
        return (i < max_iter) & (jnp.abs(k_new - k_old) > tol)

    def body_fn(val):
        k_old, _, i = val
        F_val = F(k_old, s_prev, params)

        # damped update
        k_new = (1.0 - damping) * k_old + damping * F_val

        return k_new, k_new, i + 1

    # initialize with two different values to enter loop
    k_init_next = F(k0, s_prev, params)
    k_star, _, _ = lax.while_loop(cond_fn, body_fn, (k0, k_init_next, 0))

    return k_star

# =============================================================
# One step of dynamics (vectorized)
# =============================================================
@jit
def step(s_prev: jnp.ndarray, params: JaxSystemParameters):
    k_next = solve_k_bisect(s_prev, params)
    #k_next = solve_k_iterative(s_prev, params)

    alpha = params.alpha
    gamma = params.gamma
    delta = params.delta
    E = params.E

    N = s_prev.size
    e = E / N
    xi = (1.0 - alpha) / alpha

    k_t = jnp.mean(s_prev)
    R_t = alpha * k_t ** (alpha - 1.0)
    w_t = (1 - alpha) * k_t ** alpha
    R_next = alpha * k_next ** (alpha - 1.0)
    w_next = (1 - alpha) * k_next ** alpha

    inside = R_t * s_prev + w_t + e - gamma * ((xi + 1.0) * R_t * k_t - k_next + e) - (w_next + e) / (delta * R_next)
    s_next = delta / (1 + delta) * jnp.maximum(inside, 0.0)

    return s_next, (s_next, k_next)


# =============================================================
# Simulation via lax.scan
# =============================================================
@partial(jit, static_argnames=("T",))
def simulate(s0: jnp.ndarray, params: JaxSystemParameters, T: int):
    _, (s_hist, k_hist) = lax.scan(lambda s, _: step(s, params), s0, None, length=T)
    return s_hist, k_hist


# =============================================================
# Run example
# =============================================================
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    N = 5      # number of agents
    T = 200    # number of time periods

    s0 = jax.random.uniform(key, shape=(N,), minval=0.5, maxval=1.5)
    s0 /= jnp.mean(s0)  # normalize to have mean 1.0
    s0 /= 2
    params = JaxSystemParameters(alpha=0.3, gamma=0.75, delta=4.0, E=0.0)

    # simulate
    s_path, k_path = simulate(s0, params, T)

    # --------------------------
    # Plot dynamics
    # --------------------------
    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(range(T), s_path[:, i], label=f'Agent {i+1}', alpha=0.6)
    plt.plot(range(T), k_path, "k--", linewidth=2, label="Mean k")
    plt.xlabel("Time step")
    plt.ylabel("Savings / Capital")
    plt.title("Dynamics of Savings Over Time (Optimized JIT + Binary Search)")
    plt.grid(True)
    plt.show()
