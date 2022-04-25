import numpy as np

#use the library form 5th sem computational course
def RungKutta4(f,u0,t):

    nt = len(t)

    nu = len(u0)
    u = np.zeros((nt,nu))
    u[0] = u0

    for k in range(nt-1):
        dt = t[k+1] - t[k]

        k1 = dt*f(u[k], t[k])
        k2 = dt*f(u[k]+(k1/2), t[k]+(dt/2))
        k3 = dt*f(u[k]+(k2/2),t[k]+(dt/2))
        k4 = dt*f( u[k]+k3,t[k]+dt)

        du = (k1 + 2*k2 + 2*k3 +k4)/6

        u[k+1] =u[k]+du 

    #return the increment of value with each step as an array     
    return u