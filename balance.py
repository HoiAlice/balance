import numpy as np
import jax.numpy as jnp
import pandas as pd

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
    table_new_new = np.zeros((15,59,57))
    table_new_new[:,:,:56] = table_new[:,:,:56]
    table_new_new[:,:,56] = np.sum(table_new[:,:,56:], axis = 2)
    # 56 отраслей, 3 первичных ресурса, 1 агрегированный потребитель
    tables = []
    for year in range(15):
        table = table_new_new[14]
        I = []
        J = []
        for i in range(56):
            if np.sum(table[i,:]) == 0:
                I.append(i)
            if np.sum(table[:,i]) == 0:
                J.append(i)
        table = jnp.array([[table[i, j] for j in range(57) if j not in J] for i in range(59) if i not in I]) #выкинул нулевые строки\столбцы
        table = table
        tables.append(table)
    return tables

def get_W(Z, rho):
    A = jnp.sum(Z, axis = 1)
    Z_ = jnp.transpose(jnp.transpose(Z) / A)
    W_ = jnp.power(Z_, (1+rho)/rho)
    W, W0 = W_[:,:-1], W_[:,-1]
    return W, W0

def CES(p, W, rho):
    A = jnp.transpose(W) * p
    B = jnp.power(jnp.transpose(A), rho / (1 + rho))
    q = jnp.power(jnp.sum(B, axis = 0), (1 + rho) / rho)
    return q

def JCES(p, W, rho):
    n = jnp.transpose(W).shape[0]
    return  jnp.power(jnp.divide(jnp.transpose(p * jnp.transpose(jnp.power(W, rho))),p[:n]), 1/(1+rho))
    
    