import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd

# все численные параметры жестко заданы
def read_NIOT(filepath):
    table_raw = pd.read_excel(filepath, index_col=0, sheet_name='National IO-tables').to_numpy()
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

@jax.custom_jvp
def J_to_Z(J, Z0):
    n, m = J.shape
    n, m = m, n - m
    M = jnp.diag(jnp.sum(J, axis = 0)) - J[:n,:]
    y = jnp.matmul(jnp.linalg.inv(M),Z0)
    Z_pred = y * J
    return Z_pred

@J_to_Z.defjvp
def J_to_Z_jacobian(primals, tangents):
    J, Z0 = primals
    dJ, dZ0 = tangents
    n, m = J.shape
    n, m = m, n - m
    M = jnp.diag(jnp.sum(J, axis = 0)) - J[:n,:]
    dM = jnp.diag(jnp.sum(dJ, axis = 0)) - dJ[:n,:] 
    y = jnp.matmul(jnp.linalg.inv(M), Z0)
    dy = jnp.matmul(jnp.linalg.inv(M), dZ0) - jnp.matmul(jnp.linalg.inv(M), jnp.matmul(dM, y))
    Z_pred = y * J
    dZ_pred = dy * J + y * dJ
    return (Z_pred, dZ_pred)