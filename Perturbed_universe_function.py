from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt

import RungeKutta_4 as Rk

# Parameter values
h = 0.5
Omega_m = 1

#a_eq = 0.5*(k_eq**2)*(10*h)**(-2)
a_eq = 16.6 * 10**(-5)
k_eq = np.sqrt(2*a_eq)*10*h
eta_3 = 2*np.sqrt(2)/k_eq


#x = eta/eta_3
# eta - time factor = 1/aH where H - hubbles constant at time t
#eta_3 - time factor at time at a = 3*a_eq where a_eq - scale factor at Radiation-Matter Equilibrium\

#Function defining the Einsteins equation in terms of x
def d_phi(x, k,delta_m, delta_r, Phi):


    dP = -(1/x)*(1.+x)/(1+0.5*x)*Phi - (1/(1+x))*delta_m - (1/((2*x+x**2)*(1+x)))*delta_r  - (x/3)*((1+0.5*x)/(1+x))*k**2*Phi
    
    return dP

#List of Boltzmann and Einstein equation coupled in a array as vector space

def Coupled_equations(w,x):
    delta_m, v_m, delta_r, v_r, Phi = w


    #Matter perturbation 
    A= -k*v_m + 3*d_phi(x,k, delta_m, delta_r, Phi)
    B = -(1/x)*((1+x)/(1+0.5*x))*v_m +k*Phi
    
    #raditation/Photon Perturbation
    C = -(4/3)*k*v_r +4*d_phi(x,k, delta_m, delta_r, Phi)
    D = +0.25*k*delta_r + k*Phi
    E = d_phi(x,k,delta_m, delta_r, Phi)

    coupled_eq = np.array([A,B,C,D,E])
    
    return coupled_eq

def analytical_solution(a):
    a_eq = 16.6 * 10**(-5)

    func = 0.1*((a_eq)**3)*a**(-3)*(16*((a_eq+a)/a_eq)**(0.5)+9*(a/a_eq)**3+2*(a/a_eq)**2-8*(a/a_eq)-16)  

    return func


#here k is K/eta_3
def initial_condition(a0,at,i):
    K = np.array([ h*10**(-3), h*10**(-2), h*0.05 ,1*h,10*h])
    k= eta_3*K[i]


    #parameter range
    a= np.linspace(a0,at,1000)
#converting the x term into scale factor and setting limits
    x1 = np.sqrt(1+(a/a_eq)) - 1
# INITIAL CONDITIONS
    x_in = x1[0]
# Adiabatic condition 
    delta_m0 = 1+0.25*x_in
    v_m0 = -(1/3)*k*x_in + (1/12)*k*x_in**2
    delta_r0 = 4/3 + x_in/3
    v_r0 = v_m0
    Phi0 = -(2/3)+x_in/12

    w0 = np.array([delta_m0, v_m0, delta_r0, v_r0, Phi0])

    return w0,k,x1,a


#LARGE SCALE
w1,k,x1,a = initial_condition(10**(-6),5*10**(-2),0)
Sol1 = Rk.RungKutta4(Coupled_equations, w1, x1)

w2,k,x2,a = initial_condition(10**(-6),5*10**(-2),1)
Sol2 = Rk.RungKutta4(Coupled_equations, w2, x2)

w3,k,x3,a =initial_condition(10**(-6),5*10**(-2),2)
Sol3 = Rk.RungKutta4(Coupled_equations, w3, x3)


plt.title(" Evolution of potential in a CDM model for Large scale mode")
plt.plot(a, analytical_solution(a),'--', color = 'red' ,label=f'Analytical Solution')



plt.plot(a,Sol1[:,4]/w1[4], color = 'blue', label=f'k = 0.001 h M/pc')
plt.plot(a,Sol2[:,4]/w2[4], color = 'green',label=f'k = 0.01 h M/pc')
plt.plot(a,Sol3[:,4]/w3[4], color = 'orange',label=f'k = 0.05 h M/pc')
plt.legend(loc='best')
plt.xlabel('a')
plt.ylabel('Phi/Phi(0)')
plt.vlines(a_eq,0.84,1,label='a_eq')
plt.xscale("log")
plt.show()


#SMALL SCALE 
w4,k,x4,a2 = initial_condition(10**(-7),10**(-3),3)
Sol4 = Rk.RungKutta4(Coupled_equations, w4, x4)

# due to very high value of k my RK4 library didnt work
w5,k,x5,a2 = initial_condition(10**(-7),10**(-3),4)
Sol5 =odeint(Coupled_equations, w5, x5)


plt.title(" Evolution of potential in a CDM model for Small scale mode")
plt.plot(a2,Sol4[:,4]/w4[4], color = 'blue',label=f'k = 1 h M/pc')
plt.plot(a2,Sol5[:,4]/w5[4], color = 'green',label=f'k = 10 h M/pc ')

plt.legend(loc='best')
plt.xlabel('a')
plt.ylabel('Phi/Phi(a=0)')

plt.xscale("log")
plt.show()








