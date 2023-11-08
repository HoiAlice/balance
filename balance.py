import numpy as np
import jax
from jax import numpy as jnp
import pandas as pd
from ortools.linear_solver import pywraplp

# все численные параметры жестко заданы
def read_NIOT(filepath):
    table_raw = pd.read_excel('NIOTS/RUS_NIOT_nov16.xlsx', index_col=0, sheet_name='National IO-tables').to_numpy()
    table_cutted = table_raw[1:,3:-1].reshape(15, 120, 62)
    table_new = np.zeros((15,59,62))
    table_new[:,:56,:] = table_cutted[:,:56,:] #межотраслевое в пределах страны
    table_new[:,56,:] = np.sum(table_cutted[:,56:112,:], axis = 1) + table_cutted[:,118,:] #импорт + международный транспорт
    table_new[:,57,:] = np.sum(table_cutted[:,113:117,:], axis = 1) #зарплаты
    table_new[:,58,:] = table_cutted[:,117,:] #добавленная стоимость
    # 56 отраслей, 3 первичных ресурса, 6 конечных потребителя
    tables = []
    for year in range(15):
        table = table_new[year]
        I = []
        J = []
        for i in range(56):
            if np.sum(table[i,:]) == 0:
                I.append(i)
            if np.sum(table[:,i]) == 0:
                J.append(i)
        table = jnp.array([[table[i, j] for j in range(62) if j not in J] for i in range(59) if i not in I]) #выкинул нулевые строки\столбцы
        table = table.at[-3:,-6:].set(jnp.zeros((3, 6)))
        tables.append(table)
    return tables


@jax.jit
def get_W(Z, rho): #вот тут однозначно косяк в реализации, но какой? 
    n1, n2 = Z.shape
    B = jnp.zeros((n1,))
    B = B.at[n2:].set(jnp.sum(Z, axis = 1)[n2:])
    B = B.at[:n2].set(jnp.sum(Z, axis = 0))
    Z_ = jnp.transpose(jnp.transpose(Z) / B)
    W = jnp.power(Z_, (1+rho)/rho)
    return W

@jax.jit
def CES(p, W, rho):
    A = jnp.transpose(W) * p
    B = jnp.power(jnp.transpose(A), rho / (1 + rho))
    q = jnp.power(jnp.sum(B, axis = 0), (1 + rho) / rho)
    return q


def get_prices(cost_f, n, s, eps=10E-16): #простая итерация со стартом в 0, сходится оч. быстро
    m = len(s)
    p = jnp.zeros(n+m).at[n:].set(s)
    q = jnp.zeros(n).at[:].set(cost_f(p))
    while jnp.linalg.norm(p[:n] - q) >= eps:
        p = p.at[:n].set(q)
        q = q.at[:].set(cost_f(p))
    return p
    

@jax.jit
def JCES(p, W, rho): #выплевывает транспонированный якобиан
    n = jnp.transpose(W).shape[0]
    q = CES(p, W, rho)
    return  jnp.power(jnp.transpose(jnp.divide(jnp.transpose(q * jnp.power(W, rho)),p)), 1/(1+rho))
    

def primal_J(Z, Z_hat):
    n, m = Z.shape
    n, m = m, n - m
    
    solver = pywraplp.Solver.CreateSolver("GLOP")
    x = [solver.NumVar(0.0, solver.infinity(), 'x_' + str(j+1)) for j in range(n)]
    u = [[solver.NumVar(0.0, solver.infinity(), 'u^'+str(i+1)+'_'+str(j+1)) for j in range(n)] for i in range(n+m)]
    
    for j in range(n):
        solver.Add(float(jnp.sum(Z[:,j])) * x[j] >= sum([float(Z[j,k]) * x[k] for k in range(n)]))
        for i in range(n+m):
            solver.Add(u[i][j] >= float(Z_hat[i,j]) - x[j] * float(Z[i, j]))
            solver.Add(u[i][j] >= x[j] * float(Z[i, j]) - float(Z_hat[i,j]))
    
    solver.Minimize(sum([sum(u[i]) for i in range(n+m)]))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print('not optimal')
    J = solver.Objective().Value()
    x = jnp.array([x[j].solution_value() for j in range(n)])
    u = jnp.array([[u[i][j].solution_value() for j in range(n)] for i in range(n+m)])
    return J, x, u


def dual_J(Z, Z_hat):
    n, m = Z.shape
    n, m = m, n - m
    
    solver = pywraplp.Solver.CreateSolver("GLOP")
    nu = [solver.NumVar(0.0,solver.infinity(), 'nu_'+str(k+1)) for k in range(n)]
    lam = [[solver.NumVar(-1.0, 1.0, 'lam^'+str(i+1)+'_'+str(j+1)) for j in range(n)] for i in range(n+m)]
    
    for k in range(n):
        solver.Add(sum([float(Z[j,k])*nu[j] for j in range(n)]) >= sum([float(Z[i,k]) * (nu[k] + lam[i][k]) for i in range(n+m)]))
    
    solver.Maximize(sum([sum([float(Z_hat[i,j])*lam[i][j] for j in range(n)]) for i in range(n+m)]))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print('not optimal')
    J = solver.Objective().Value()
    nu = jnp.array([nu[k].solution_value() for k in range(n)])
    lam = jnp.array([[lam[i][j].solution_value() for j in range(n)] for i in range(n+m)])
    return J, nu, lam

@jax.jit
def lagrangian_J(Z, Z_hat, x, u, nu, lam):
    n, m = Z.shape
    n, m = m, n - m
    L = (jnp.trace(u @ jnp.transpose(1 - lam)) + jnp.trace(Z_hat @ jnp.transpose(lam)) +
    jnp.dot(nu, jnp.matmul(Z[:33,:], x)) - jnp.sum(x * nu * Z) - jnp.sum(x * Z * lam)) 
    return L

def grad_J(Z, Z_hat):
    J1, x, u = primal_J(Z, Z_hat)
    J2, nu, lam = dual_J(Z, Z_hat)
    grad = jax.grad(lagrangian_J)(Z, Z_hat, x, u, nu, lam)
    return grad
    