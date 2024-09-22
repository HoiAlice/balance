import jax
from jax import numpy as jnp

@jax.jit
def weights(Z, rho):
    W = jnp.power(Z/jnp.sum(Z, axis = 0), (1+rho)/rho)
    return W

@jax.custom_jvp
@jax.jit
def costs(p, W, rho):
    A = jnp.transpose(W) * p
    B = jnp.power(jnp.transpose(A), rho / (1 + rho))
    q = jnp.power(jnp.sum(B, axis = 0), (1 + rho) / rho)
    return q

# нужно будет учесть dW, drho
@costs.defjvp
def costs_jacobian(primals, tangents):
    p, W, rho = primals
    dp, dW, drho = tangents
    f = costs(p, W, rho)
    df = jnp.transpose(proportions(p, W, rho)) @ dp
    return (f, df)

@jax.custom_jvp
@jax.jit
def proportions(p, W, rho): 
    n = jnp.transpose(W).shape[0]
    q = costs(p, W, rho)
    return  jnp.power(jnp.transpose(jnp.divide(jnp.transpose(q * jnp.power(W, rho)),p)), 1/(1+rho))

def prop_grad(p, W, rho, f, dp, dW, drho, df):
    return jnp.nan_to_num(drho * (jnp.log(W) + jnp.log(p) - jnp.log(f)) / (1 + rho) ** 2 + 
                   df / (1 + rho) / f - dp / (1 + rho) / p + dW * rho / (1 + rho) / W)

@proportions.defjvp
def proportions_jacobian(primals, tangents):
    p, W, rho = primals
    dp, dW, drho = tangents
    n, m = W.shape
    
    f, df = jax.jvp(costs, primals, tangents)
    J = proportions(p, W, rho) 
    dJ = jax.vmap(prop_grad)(jnp.kron(p,jnp.ones((m,))), 
                             W.reshape(-1), 
                             jnp.kron(jnp.ones((n,)),rho), 
                             jnp.kron(jnp.ones((n,)),f),
                             jnp.kron(dp,jnp.ones((m,))),
                             dW.reshape(-1), 
                             jnp.kron(jnp.ones((n,)),drho),
                             jnp.kron(jnp.ones((n,)),df)).reshape(n, m)
    return (J, dJ)

#изучи сходимость получше
@jax.custom_jvp
def balance_prices(W, rho, s, h):
    n, m = W.shape
    n, m = m, n - m
    p = (jnp.ones(n+m) * jnp.min(s)).at[n:].set(s)
    q = (jnp.ones(n) * jnp.min(s)).at[:].set(costs(p, W, rho) + h)
    while jnp.linalg.norm(p[:n] - q) >= 10e-16:
        p = p.at[:n].set(q)
        q = q.at[:].set(costs(p, W, rho) + h)
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
    dp = dp.at[:n].set(jnp.linalg.inv(jnp.eye(n) - jnp.transpose(proportions(p, W, rho))[:,:n]) @ (jax.jvp(costs, [p, W, rho], [dp, dW, drho])[1] + dh))
    return (p, dp)