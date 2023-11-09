import jax
from jax import numpy as jnp


@jax.jit
def W(Z, rho):
    W = jnp.power(Z/jnp.sum(Z, axis = 0), (1+rho)/rho)
    return W


@jax.jit
def f(p, W, rho):
    A = jnp.transpose(W) * p
    B = jnp.power(jnp.transpose(A), rho / (1 + rho))
    q = jnp.power(jnp.sum(B, axis = 0), (1 + rho) / rho)
    return q


@jax.jit
def J(p, W, rho): #выплевывает транспонированный якобиан.
    n = jnp.transpose(W).shape[0]
    q = f(p, W, rho)
    return  jnp.power(jnp.transpose(jnp.divide(jnp.transpose(q * jnp.power(W, rho)),p)), 1/(1+rho))


@jax.custom_jvp
def balance_prices(W, rho, s, h):
    n, m = W.shape
    n, m = m, n - m
    p = jnp.zeros(n+m).at[n:].set(s)
    q = jnp.zeros(n).at[:].set(f(p, W, rho) + h)
    while jnp.linalg.norm(p[:n] - q) >= 10E-8:
        p = p.at[:n].set(q)
        q = q.at[:].set(f(p, W, rho) + h)
    return p


@balance_prices.defjvp
def balance_prices_jacobian(primals, tangents):
    W, rho, s, h = primals
    dW, drho, ds, dh = tangents
    n, m = W.shape
    n, m = m, n - m
    p = balance_prices(W, rho, s, h)
    dp = jnp.zeros((n+m,))
    dp = dp.at[n:].set(ds)
    dp = dp.at[:n].set(jnp.linalg.inv(jnp.eye(n) - jnp.transpose(J(p, W, rho))[:,:n]) @ (jax.jvp(f, [p, W, rho], [dp, dW, drho])[1] + dh))
    return (p, dp)