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
def get_W(Z, Z0, rho): #тут косяк
    n, m = Z.shape
    n, m = m, n - m
    
    W = jnp.transpose(jnp.array([[jnp.power(Z[i, j]/jnp.sum(Z[:,j]),(1+rho[j])/rho[j]) for i in range(n+m)] for j in range(n)]))
    return W

@jax.jit
def CES(p, W, rho): #или тут косяк
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
def JCES(p, W, rho): #выплевывает транспонированный якобиан.
    n = jnp.transpose(W).shape[0]
    q = CES(p, W, rho)
    return  jnp.power(jnp.transpose(jnp.divide(jnp.transpose(q * jnp.power(W, rho)),p)), 1/(1+rho))