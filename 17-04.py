import numpy as np
from matplotlib import pyplot as plt

# Imposta la precisione a 16 cifre decimali
np.set_printoptions(precision=16, suppress=True)

## DEFINING THE CONSTANTS ##

R = 10  #resistance [ohm]
C = 0.0025 #capacitance [F]
L = 1   #inductance [H]

dE = lambda t: 0.2*np.sin(2.5*t)    #function of the derivative of the electrical field

## ---------------------------------------------------------------------- ##
## ONE-STEP RESOLUTION METHODS

# In this part of the code several one-step resolution methods will be applied to solve the problem
# A stability and consistency study will be conducted in order to evaluate which method fits better the problem

## ---------------------------------------------------------------------- ##

A = np.array([[0, 1],[-400,-10]])     #matrix dependant on i, i' if recasting the problem as a first-order ODE system
B = lambda F: np.array([0.0,F])     #matrix of the terms time-dependent of the problem recasted as a first-order ODE system (maybe not the best way?)
y0 = [0,0.1]    #initial values
I2 = np.eye(2)

## EXPLICIT EULER
def EE(x0, nodes):
    # nodes: number of points in which to evaluate i  

    time = np.linspace(0,5,nodes)   #vector of time
    dt = time[1]-time[0]    #increment step
    x = np.zeros((2, nodes))    #null vector containing two rows: [0] for i; [1] for i'
    x[:,0] = x0
    cee = (I2 + dt*A)   #amplification matrix
    for i in range(1, nodes):
        x[:,i] = cee @ x[:,i-1] + dt*B(dE((i-1)*dt))

    return x, cee, time

## RK2 (with alfa = 1/2, hence modified euler)
def RK2(x0, nodes):
    # nodes: number of points in which to evaluate i 
 
    time = np.linspace(0,5,nodes)   #vector of time
    dt = time[1]-time[0]    #increment step
    x = np.zeros((2, nodes))    
    x[:,0] = x0
    crk = I2 + dt*(A@(I2 + (dt/2)*A))      #amplification matrix
    for i in range(1, nodes):
        x[:,i] = crk @ x[:,i-1] + dt*(B(dE((i-0.5)*dt)) + (dt/2)*(A@B(dE((i-1)*dt))))
    
    return x, crk

## CRANK-NICHOLSON
def CN(x0, nodes):
    # nodes: number of points in which to evaluate i 
       
    time = np.linspace(0,5,nodes)   #vector of time
    dt = time[1]-time[0]    #increment step
    x = np.zeros((2, nodes))    
    x[:,0] = x0
    a = I2 - (dt/2)*A
    b = I2 + (dt/2)*A
    ccn = np.linalg.solve(a,b)  #amplification matrix
    for i in range(1, nodes):
        x[:,i] = ccn @ x[:,i-1] + (dt/2)*np.linalg.solve(a, (B(dE(i*dt)) + B(dE((i-1)*dt))))
    
    return x, ccn

y_ee, C_ee, time = EE(y0,250)
y_rk2, C_rks = RK2(y0,250)
y_cn, C_cn = CN(y0,250)
dt = time[1]-time[0]

## PLOTS OF i WITH THE DIFFERENT METHODS ##
plt.figure(1, figsize=(12,5))
plt.plot(time, y_ee[0,:], 'b-', linewidth = 3, label = "EE")
plt.plot(time, y_rk2[0,:], 'r--', linewidth = 2, label = "RK2")
plt.plot(time, y_cn[0,:], 'g-', label = "CN")
plt.legend(loc = 'best')
plt.xlabel('t(s)')
plt.ylabel('i(A)')
plt.title('current with one-step methods for time step:' + str(dt) +'s')
plt.grid(True)
plt.show()


## ---------------------------------------------------------------------- ##
## NEWMARK RESOLUTION METHODS

# In this part of the code some Newmark resolution methods will be applied to solve the problem
# A stability and consistency study will be conducted in order to evaluate which method fits better the problem

## ---------------------------------------------------------------------- ##

def verlet(x0, nodes):
    # nodes: number of points in which to evaluate i 
       
    time = np.linspace(0,5,nodes)   #vector of time
    dt = time[1]-time[0]    #increment step
    y = np.zeros(nodes)
    yprime = np.zeros(nodes)
    y[0] = x0[0]
    yprime[0] = x0[1]

    for i in range(1,nodes):
        b = B(dE((i-1)*dt))    #function dependent on time evaluated at the actual point
        b2 = B(dE(i*dt))    #function dependent on time evaluated at the next point
        y[i] = (1-200*(dt**2))*y[i-1] + dt*(1-5*dt)*yprime[i-1] + (dt**2/2)*b[1]
        yprime[i] = (((1-5*dt))*yprime[i-1] - 200*dt*y[i-1] + (dt/2)*b[1] - 200*dt*y[i] + (dt/2)*b2[1])/(1+5*dt)

    return y, yprime

i_ver, ip_ver = verlet(y0, 250)

## PLOTS
plt.figure(2, figsize=(12,5))
plt.plot(time, i_ver, 'y-', label = "Verlet")
plt.legend(loc = 'best')
plt.xlabel('t(s)')
plt.ylabel('i(A)')
plt.title('current with verlet method for time step:' + str(dt) +'s')
plt.grid(True)
plt.show()

## ---------------------------------------------------------------------- ##
## CONSISTENCY STUDY

# In this part of the code some a consistency study is developed
# To do the stability study, we evaluate the relative error
# In order to do that we consider the reference solution as the one obtained with the CN method (the most stable and precise) with a very small deltat (say with 1000 nodes ==- t=0.005s)
# We will compare the current at 5s (i.e. the last value of the vector containing the currents)
# Than we compare, for different time intervals, the norm of the relative errors. the time intervals will be taken from 0.005 to 0.02 s (in order to consider the EE stability radius)
# Lastly we plot (on a log-log graph) the evolution of the error in function of the time interval. The slope of the line should be an approximation of the order of convergence (found with polyfit)

## ---------------------------------------------------------------------- ##

# getting the reference solution
sol = CN(y0, 5000)
ref_sol = sol[0][0,-1]

nodes = np.linspace(2000,500,100)
deltat = np.zeros(len(nodes))
relerr_ee = np.zeros(len(deltat))
relerr_rk2 = np.zeros(len(deltat))
relerr_cn = np.zeros(len(deltat))
relerr_ver = np.zeros(len(deltat))

for i in range(0, len(deltat)):
    deltat[i] = 5/int(nodes[i])

    sol_EE = EE(y0, int(nodes[i]))
    relerr_ee[i] = np.abs((sol_EE[0][0,-1]-ref_sol)/ref_sol)
    sol_rk2 = RK2(y0, int(nodes[i]))
    relerr_rk2[i] = np.abs((sol_rk2[0][0,-1]-ref_sol)/ref_sol)
    sol_CN = CN(y0, int(nodes[i]))
    relerr_cn[i] = np.abs((sol_CN[0][0,-1]-ref_sol)/ref_sol)
    sol_ver = verlet(y0, int(nodes[i]))
    relerr_ver[i] = np.abs((sol_ver[0][-1]-ref_sol)/ref_sol)

# ORDER OF CONVERGENCE ESTIMATION

p_ee = np.polyfit(np.log(deltat),np.log(relerr_ee),1)
p_rk2 = np.polyfit(np.log(deltat),np.log(relerr_rk2),1)
p_cn = np.polyfit(np.log(deltat),np.log(relerr_cn),1)
p_ver = np.polyfit(np.log(deltat),np.log(relerr_ver),1)

#plots

plt.figure(3, figsize=(12,5))
plt.loglog(deltat, relerr_ee, 'b-', label = 'EE, p = '+str(p_ee[0]))
plt.loglog(deltat, relerr_rk2, 'r-', label = 'RK2, p = '+str(p_rk2[0]))
plt.loglog(deltat, relerr_cn, 'g-', label = 'CN, p = '+str(p_cn[0]))
plt.loglog(deltat, relerr_ver, 'y-', label = 'VERLET, p = '+str(p_ver[0]))

plt.legend(loc ='best')
plt.xlabel('Delta t')
plt.ylabel('Relative error')
plt.title('Consistency study')
plt.grid(True, which = 'both')
plt.show()