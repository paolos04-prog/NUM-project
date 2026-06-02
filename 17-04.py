import numpy as np
from matplotlib import pyplot as plt

# Setting the precision to 16 decimals
np.set_printoptions(precision=16, suppress=True)

## DEFINING THE CONSTANTS ##

R = 10  #resistance [ohm]
C = 0.0025 #capacitance [F]
L = 1   #inductance [H]

approxdt = 0.001   #approximate time step to insert by hand
tM = 5      # maximum time of the simulation
nd = round(tM/approxdt) + 1  #nodes (int) corresponding to approxdt

dE = lambda t: 0.2*np.sin(2.5*t)    #function of the derivative of the electrical field

to_run = 'part2'     # string to change to run different parts of the code: 
# 'solution' compares different methods for solving the ivp problem
# 'stability' implements the stability study
# 'consistency' implements the consistency study
# 'part2' to run the part of the code concerning the non-linear problem

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
    # REMARK: for this method, the stability radius (easily also to calculate by hand) is dt = 0.025s, that corresponds to 200 nodes

    time = np.linspace(0,tM,nodes)   #vector of time
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
 
    time = np.linspace(0,tM,nodes)   #vector of time
    dt = time[1]-time[0]    #increment step
    x = np.zeros((2, nodes))    
    x[:,0] = x0
    crk = I2 + dt*(A@(I2 + (dt/2)*A))      #amplification matrix
    for i in range(1, nodes):
        x[:,i] = crk @ x[:,i-1] + dt*(B(dE((i-0.5)*dt)) + (dt/2)*(A@B(dE((i-1)*dt))))
    
    return x, crk

def RK4(x0,nodes):
    # nodes: number of points in which to evaluate i 
 
    time = np.linspace(0,tM,nodes)   #vector of time
    dt = time[1]-time[0]    #increment step
    x = np.zeros((2, nodes))    
    x[:,0] = x0
    # use standard RK4: compute K1..K4
    for i in range(1, nodes):
        t_n = (i-1)*dt
        K1 = A @ x[:,i-1] + B(dE(t_n))
        K2 = A @ (x[:,i-1] + (dt/2)*K1) + B(dE(t_n + dt/2))
        K3 = A @ (x[:,i-1] + (dt/2)*K2) + B(dE(t_n + dt/2))
        K4 = A @ (x[:,i-1] + dt*K3) + B(dE(t_n + dt))
        x[:,i] = x[:,i-1] + dt/6*(K1 + 2*K2 + 2*K3 + K4)
    # to obtain the amplification matrix, we neglect the terms that arn't multiplying the unknown vector
    # in other words, we study the homogenous case
    K1_matrix = A
    K2_matrix = A + 0.5 * dt * A @ A
    K3_matrix = A + 0.5 * dt * A @ A + 0.25 * dt**2 * (A @ A @ A)
    K4_matrix = A + dt * A @ A + 0.5 * dt**2 * (A @ A @ A) + 0.25 * dt**3 * (A @ A @ A @ A)

   
    crk4 = I2 + dt/6*(K1_matrix + 2*K2_matrix + 2*K3_matrix + K4_matrix)      #amplification matrix
    #crk4 = I2 + dt*A + (dt*A)**2/2 + (dt*A)**3/6 + (dt*A)**4/24
    return x, crk4

## CRANK-NICHOLSON
def CN(x0, nodes):
    # nodes: number of points in which to evaluate i 
       
    time = np.linspace(0,tM,nodes)   #vector of time
    dt = time[1]-time[0]    #increment step
    x = np.zeros((2, nodes))    
    x[:,0] = x0
    a = I2 - (dt/2)*A
    b = I2 + (dt/2)*A
    ccn = np.linalg.solve(a,b)  #amplification matrix
    for i in range(1, nodes):
        x[:,i] = ccn @ x[:,i-1] + (dt/2)*np.linalg.solve(a, (B(dE(i*dt)) + B(dE((i-1)*dt))))
    
    return x, ccn

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
        #amplification matrix for the verlet method (Found using hand computation)
        C_verlet = np.array([[1 - 200*dt**2, (1-5*dt)*dt], [-200*dt*(2-200*dt**2)/(1+5*dt), (1000*dt**3 - 200*dt**2 -5*dt + 1)/(1+5*dt)]])

    return y, yprime, C_verlet


## ---------------------------------------------------------------------- ##
## STUDY OF THE STABILITY

# In this part of the code we sill study the stability of the different numerical scheme we implemented previously

## ---------------------------------------------------------------------- ##
##Compute the eigenvalues of the amplification matrix to check the stability
def eigenvalues(ampli_matrix):
    eigen_val, eigen_vect = np.linalg.eig(ampli_matrix)
    return eigen_val, eigen_vect

#Stability
def stability(eig_val):
    stable = True
    additional_check = False
    for eigenval in eig_val:
        norm_eig_val = np.abs(eigenval)
    
        if norm_eig_val > 1:
            stable = False
        elif norm_eig_val == 1:
            stable  = False
            additional_check = True
    return stable, additional_check

def stable_text(eig_val):  
    stable = True
    additional_check = False
    for eigenval in eig_val:
        norm_eig_val = np.abs(eigenval)
    
        if norm_eig_val > 1:
            stable = False
        elif norm_eig_val == 1:
            stable  = False
            additional_check = True

    if stable == True:
        return " is stable"
    elif stable == False and additional_check == False:
        return " is unstable"
    elif stable == False and additional_check == True:
        return ", we cannot conclude on the stability because at least the absolute value of one eigenvalue is 1."
        
    #Be careful, if the eigenvalues is equal to 1, we need additional information

#Compute the radius of stability of a method
precision = 1000
def radius_stability( method,x0, precision = precision):
    """
    INPUTS
    Precision is an int, the default value is 1000

    method is a string:
        put EE, if the method used is euler explicite
        put RK2, if the method used is Runge Kutta 2
        put RK4, if the method used is Runge Kutta 4
        put CN, if the method used is Crank Nicolson
        put verlet, if the method used is Verlet
    
    x0 is an array with the initial conditions of the problem
    """

    stable = False
    i = 2
        
    while stable == False and  i < precision + 1:
        if method == "EE":
            _, C_ee, time = EE(x0,i)
            eigvalue = eigenvalues(C_ee)[0]
            

        elif method == "RK2":
            _, C_rks = RK2(x0,i)
            eigvalue = eigenvalues(C_rks)[0]

        elif method == "RK4":
            _, C_rks4 = RK4(x0,i)
            eigvalue = eigenvalues(C_rks4)[0]
            

        elif method == "CN":
            _, C_cn = CN(x0,i)
            eigvalue = eigenvalues(C_cn)[0]

        elif method == "verlet":
            _, _ , C_verlet = verlet(x0,i)
            eigvalue = eigenvalues(C_verlet)[0]
            
        else:
            print("The method you entered is not valid, refer to the instructions for the possible methods")

        stable, add_check = stability(eigvalue)
        i += 1
        
    
    time = np.linspace(0, tM, i - 1)
    dt  =  time[1]-time[0]
    if stable == True:
        if i == 3:
            print(f"The {method} method is unconditionally stable")
        else:
            print(f"The {method} method has a radius of convergence of {dt}")
    elif stable == False and add_check == False:
        print(f"The {method} method is unconditionally unstable for a time step of {dt}")
    elif stable == False and add_check == True:
        print(f"For the {method} method no radius of convergence was found, for a time step of {dt} and at least the absolute value of one eigenvalue is 1.")

match to_run:

    case 'solution':
        y_ee, C_ee, time = EE(y0,nd)       #200 is the number of nodes for which EE is stable, hence also all the other schemes are stable, since they have a stab radius greater than EE
        #y_rk2, C_rks = RK2(y0,nd)
        #y_rk4, C_rks = RK4(y0,nd)
        y_cn, C_cn = CN(y0,nd)
        i_ver, ip_ver, C_ver = verlet(y0, nd)
        dt = time[1]-time[0]

        ## PLOTS OF i WITH THE DIFFERENT METHODS ##
        plt.figure(figsize=(12,5))
        plt.plot(time, y_ee[0,:], 'b-', linewidth = 3, label = "EE")
        #plt.plot(time, y_rk2[0,:], 'r--', linewidth = 2, label = "RK2")
        #plt.plot(time, y_rk4[0,:], 'm-', linewidth = 2, label = "RK4")
        plt.plot(time, y_cn[0,:], 'g-.', linewidth = 2, label = "CN")
        plt.plot(time, i_ver, 'y-', label = "Verlet")
        plt.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
        plt.legend(loc = 'best')
        plt.xlabel('t(s)')
        plt.ylabel('i(A)')
        plt.title('current with different methods for time step:' + f"{dt:.4e}" + 's')
        plt.grid(True)
        plt.show()

    case 'stability':
        nodes = nd
        y_ee, C_ee, time = EE(y0,nodes)
        y_rk2, C_rks = RK2(y0,nodes)
        y_rk4, C_rks4 = RK4(y0,nodes)
        y_cn, C_cn = CN(y0,nodes)
        i_ver, ip_ver, C_verlet = verlet(y0, nodes)
        dt = time[1]-time[0]

        eigenvalue_ee = eigenvalues(C_ee)[0]
        eigenvalue_rks = eigenvalues(C_rks)[0]
        eigenvalue_rks4 = eigenvalues(C_rks4)[0]
        eigenvalue_cn = eigenvalues(C_cn)[0]
        eigenvalue_verlet = eigenvalues(C_verlet)[0]


        stable_ee = stable_text(eigenvalue_ee)
        stable_rks = stable_text(eigenvalue_rks)
        stable_rks4 = stable_text(eigenvalue_rks4)
        stable_cn = stable_text(eigenvalue_cn)
        stable_verlet = stable_text(eigenvalue_verlet)


        #check if the method is stable or not for a given time step dt and gives its radius of convergence
        print(f"for a time step of" + f"{dt:.4e}" + "s:")
        print(f"The Explicit Euler method{stable_ee}")
        radius_stability("EE", y0)
        print(f"The RK2 method{stable_rks}")
        radius_stability("RK2", y0)
        print(f"The RK4 method{stable_rks4}")
        radius_stability("RK4", y0)
        print(f"The Crank Nicolson method{stable_cn}")
        radius_stability("CN", y0)
        print(f"The verlet method{stable_verlet}")
        radius_stability("verlet", y0)

    case 'consistency':
        ## ---------------------------------------------------------------------- ##
        ## CONSISTENCY STUDY

        # In this part of the code a consistency study is developed, evaluating the relative error
        # In order to do that we consider the reference solution as the one obtained with the CN method (the most stable and precise) with a very small dt (say with 1000 nodes ==- t=0.005s)
        # We will compare the current at 5s (i.e. the last value of the vector containing the currents)
        # Than we compare, for different time intervals, the norm of the relative errors. the time intervals will be taken from 0.005 to 0.02 s (in order to consider the EE stability radius)
        # Lastly we plot (on a log-log graph) the evolution of the error in function of the time interval. The slope of the line should be an approximation of the order of convergence (found with polyfit)

        ## ---------------------------------------------------------------------- ##

        # getting the reference solution
        sol = CN(y0, 200000)
        ref_sol = sol[0][0,-1]

        nodes = np.linspace(2000,500,100)
        deltat = np.zeros(len(nodes))
        relerr_ee = np.zeros(len(deltat))
        #relerr_rk2 = np.zeros(len(deltat))
        relerr_rk4 = np.zeros(len(deltat))
        relerr_cn = np.zeros(len(deltat))
        relerr_ver = np.zeros(len(deltat))

        for i in range(0, len(deltat)):
            deltat[i] = tM/int(nodes[i])

            sol_EE = EE(y0, int(nodes[i]))
            relerr_ee[i] = np.abs((sol_EE[0][0,-1]-ref_sol)/ref_sol)
            sol_rk4 = RK4(y0, int(nodes[i]))
            relerr_rk4[i] = np.abs((sol_rk4[0][0,-1]-ref_sol)/ref_sol)
            #sol_rk2 = RK2(y0, int(nodes[i]))
            #relerr_rk2[i] = np.abs((sol_rk2[0][0,-1]-ref_sol)/ref_sol)
            sol_CN = CN(y0, int(nodes[i]))
            relerr_cn[i] = np.abs((sol_CN[0][0,-1]-ref_sol)/ref_sol)
            sol_ver = verlet(y0, int(nodes[i]))
            relerr_ver[i] = np.abs((sol_ver[0][-1]-ref_sol)/ref_sol)

        # ORDER OF CONVERGENCE ESTIMATION

        p_ee = np.polyfit(np.log(deltat),np.log(relerr_ee),1)
        #p_rk2 = np.polyfit(np.log(deltat),np.log(relerr_rk2),1)
        p_rk4 = np.polyfit(np.log(deltat),np.log(relerr_rk4),1)
        p_cn = np.polyfit(np.log(deltat),np.log(relerr_cn),1)
        p_ver = np.polyfit(np.log(deltat),np.log(relerr_ver),1)

        #plots

        plt.figure(figsize=(12,5))
        plt.loglog(deltat, relerr_ee, 'b-', label = 'EE, p = '+str(p_ee[0]))
        #plt.loglog(deltat, relerr_rk2, 'r-', label = 'RK2, p = '+str(p_rk2[0]))
        plt.loglog(deltat, relerr_rk4, 'm-', label = 'RK4, p = '+str(p_rk4[0]))
        plt.loglog(deltat, relerr_cn, 'g-', label = 'CN, p = '+str(p_cn[0]))
        plt.loglog(deltat, relerr_ver, 'y-', label = 'VERLET, p = '+str(p_ver[0]))

        plt.legend(loc ='best')
        plt.xlabel('Delta t')
        plt.ylabel('Relative error')
        plt.title('Consistency study')
        plt.grid(True, which = 'both')
        plt.show()

    case 'part2':
        ## ---------------------------------------------------------------------- ##
        ## SECOND PART, NON LINEAR PROBLEM

        # In this part of the code the second part of the problem is developed
        # This consists in studying a more realistic version of the problem, described by a non-linear equation
        # The problem will be tackled with different strategies: firstly with a fully explicit scheme, then with some implicit schemes (with different initialization approaches)

        ## ---------------------------------------------------------------------- ##

        to_run2 = 'consistency2'    #mathc-case for the second part. 
        # 'solution2' to get the sol with different methods
        # 'consistency2' to get the consistency study

        eps = 1e-7
        def B2(x,F):    #function F(t,x(t)) for the new non-linear problem
            y = x[0]
            yround = np.sqrt(y**2 + eps)
            yp = x[1]
            return np.array([yp, -1e4*yround*yp- 400*y + F])
        
        def EE2(x0, nodes):
            # nodes: number of points in which to evaluate i 
       
            time = np.linspace(0,tM,nodes)   #vector of time
            dt = time[1]-time[0]    #increment step
            x = np.zeros((2, nodes))    
            x[:,0] = x0
            
            for i in range(1, nodes):
                x[:,i] = x[:,i-1] + dt*B2(x[:,i-1], dE((i-1)*dt))

            return x, time
        
        def RK42(x0, nodes):
            # nodes: number of points in which to evaluate i 
    
            time = np.linspace(0,tM,nodes)   #vector of time
            dt = time[1]-time[0]    #increment step
            x = np.zeros((2, nodes))    
            x[:,0] = x0
            # Computing K1..K4 for the nonlinear system
            for i in range(1, nodes):
                t_n = (i-1)*dt
                K1 = B2(x[:,i-1], dE(t_n))
                K2 = B2(x[:,i-1] + (dt/2)*K1, dE(t_n + dt/2))
                K3 = B2(x[:,i-1] + (dt/2)*K2, dE(t_n + dt/2))
                K4 = B2(x[:,i-1] + dt*K3, dE(t_n + dt))
                x[:,i] = x[:,i-1] + dt/6*(K1 + 2*K2 + 2*K3 + K4)

            return x

        #REMARK: Now we will implement the crank-nicolson scheme (implicit) with different strategies: 
        # explicit method initialization (EE/RK2) or fixed-point/NR iterations
        
        def CN2(x0, nodes):     
            # nodes: number of points in which to evaluate i 
       
            time = np.linspace(0,tM,nodes)   #vector of time
            dt = time[1]-time[0]    #increment step
            x = np.zeros((2, nodes))    
            x[:,0] = x0

            for i in range(1, nodes):
                guess = x[:,i-1] + dt*B2(x[:,i-1], dE((i-1)*dt))    #EE predictor step
                #guess = x[:,i-1] + dt*B2(x[:,i-1]+(dt/2)*B2(x[:,i-1],dE((i-1)*dt)) , dE((i-0.5)*dt))    #rk2 predictor step
                x[:,i] = x[:,i-1] + (dt/2)*(B2(x[:,i-1], dE((i-1)*dt)) + B2(guess, dE(i*dt)))   #corrector step

            return x
        
        def CN2fp(x0, nodes, tol):    #fixed point iterations with while loop, tolerance
            # nodes: number of points in which to evaluate i 
       
            time = np.linspace(0,tM,nodes)   #vector of time
            dt = time[1]-time[0]    #increment step
            x = np.zeros((2, nodes))    
            x[:,0] = x0
            nmax = 0

            for i in range(1, nodes):
                
                n = 0
                guess = x[:,i-1] + dt*B2(x[:,i-1], dE((i-1)*dt))    #EE predictor step
                #guess = x[:,i-1] + dt*B2(x[:,i-1]+(dt/2)*B2(x[:,i-1],dE((i-1)*dt)) , dE((i-0.5)*dt))    #rk2 predictor step
                guessk = x[:,i-1] + (dt/2)*(B2(x[:,i-1], dE((i-1)*dt)) + B2(guess, dE(i*dt)))   #corrector step
                err = np.linalg.norm(guess-guessk)

                while err > tol:        #fixed point iterations
                    guess = guessk
                    guessk = x[:,i-1] + (dt/2)*(B2(x[:,i-1], dE((i-1)*dt)) + B2(guess, dE(i*dt)))   
                    err = np.linalg.norm(guess-guessk)
                    n = n+1
                
                x[:,i] = guessk

                if n > nmax:
                    nmax = n
            
            print('fixed point iterations with maximum number of iterations: ', nmax)
            return x
        
        def CN2nr(x0, nodes, tol):    #newton raphson iterations with while loop, tolerance
            # nodes: number of points in which to evaluate i 
       
            time = np.linspace(0,tM,nodes)   #vector of time
            dt = time[1]-time[0]    #increment step
            x = np.zeros((2, nodes))    
            x[:,0] = x0
            nmax = 0

            for i in range(1, nodes):
                
                n = 0
                guess = x[:,i-1] + dt*B2(x[:,i-1], dE((i-1)*dt))    #EE predictor step
                #guess = x[:,i-1]
                #guess = x[:,i-1] + dt*B2(x[:,i-1]+(dt/2)*B2(x[:,i-1],dE((i-1)*dt)) , dE((i-0.5)*dt))    #rk2 predictor step
                guessk = x[:,i-1] + (dt/2)*(B2(x[:,i-1], dE((i-1)*dt)) + B2(guess, dE(i*dt)))   #corrector step
                err = np.linalg.norm(guess-guessk)

                while err > tol and n < 100:        #nr iterations
                    guess = guessk
                    G =  guess - x[:,i-1] - (dt/2)*(B2(x[:,i-1], dE((i-1)*dt)) + B2(guess, dE(i*dt)))      #cn function, of which is necessary to compute the jacobian
                    ik = guess[0]   #current k
                    ipk = guess[1]  #current' k
                    jg = np.array([[1, -dt/2], [dt*5000*ipk*(ik/np.sqrt(ik**2 + eps))+200*dt, 1+dt*5000*np.sqrt(ik**2 + eps)]])   #jacobian matrix
                    deltaguess = np.linalg.solve(jg, -G)    #guessk-guess
                    guessk = guess + deltaguess     #find the new guessk
                    err = np.linalg.norm(deltaguess)
                    n = n+1
                
                x[:,i] = guessk

                if n > nmax:
                    nmax = n
            
            print('newton raphson iterations with maximum number of iterations: ', nmax)
            return x

        match to_run2:
            
            case 'solution2':
                y_ee2, time = EE2(y0, nd)
                #y_cne = CN2(y0, nd)
                y_rk42 = RK42(y0, nd)
                #y_cnfp = CN2fp(y0, nd, 1e-5)
                y_cnnr = CN2nr(y0, nd, 1e-5)
                dt = time[1] - time[0]
                dtr = np.round(dt,8)
                
                plt.figure(figsize=(15,10))
                
                #EE
                plt.subplot(1,3,1)
                plt.plot(time, y_ee2[0], 'b')
                plt.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)      #to highlight i = 0A
                plt.xlabel('time (s)'); plt.ylabel('current (A)'); plt.title('Explicit Euler'); plt.grid(True)
                
                #CN with EE guess
                #plt.subplot(2,3,2)
                #plt.plot(time, y_cne[0])
                #plt.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)      
                #plt.xlabel('time (s)'); plt.ylabel('current (A)'); plt.title('CN with EE guess'); plt.grid(True)
                
                #CN with EE initial guess, fixed point iterations
                #plt.subplot(2,3,3)
                #plt.plot(time, y_cnfp[0])
                #plt.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
                #plt.xlabel('time (s)'); plt.ylabel('current (A)'); plt.title('CN with EE initial guess and FP iterations'); plt.grid(True)

                #CN with EE initial guess, newton raphson iterations
                plt.subplot(1,3,2)
                plt.plot(time, y_cnnr[0], 'g')
                plt.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
                plt.xlabel('time (s)'); plt.ylabel('current (A)'); plt.title('CN with EE initial guess and NR iterations'); plt.grid(True)

                #RK4
                plt.subplot(1,3,3)
                plt.plot(time, y_rk42[0], 'y')
                plt.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
                plt.xlabel('time (s)'); plt.ylabel('current (A)'); plt.title('RK4'); plt.grid(True)

                plt.suptitle('Comparison of Numerical Schemes for the non-linear problem, (dt =' + f"{dt:.4e}" + 's)', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()

            case 'consistency2':

                tolerance = 1e-15
                
                #getting the 'exact' solution
                ex_sol = RK42(y0, 200000)[0][-1]
                deltastep = np.linspace(9000, 15000, 50)      #vector containing nodes starting from 1k nodes to 40k nodes
                deltastept = np.zeros(len(deltastep))           #vector containing time steps deriving from the choosen nodes
                rerr_ee = np.zeros(len(deltastep))
                rerr_rk4 = np.zeros(len(deltastep))
                #rerr_cne = np.zeros(len(deltastep))
                #rerr_cnfp = np.zeros(len(deltastep))
                rerr_cnnr = np.zeros(len(deltastep))
                
                def cons(x0, n, tol):
                    y_ee2 = EE2(x0, n)
                    y_rk4 = RK42(x0, n)
                    #y_cne = CN2(x0, n)
                    #y_cnfp = CN2fp(x0, n, tol)
                    y_cnnr = CN2nr(x0, n, tol)

                    error_ee = np.abs((y_ee2[0][0,-1]-ex_sol)/ex_sol)
                    error_rk4 = np.abs((y_rk4[0][-1]-ex_sol)/ex_sol)
                    #error_cne = np.abs((y_cne[0][-1]-ex_sol)/ex_sol)
                    #error_cnfp = np.abs((y_cnfp[0][-1]-ex_sol)/ex_sol)
                    error_cnnr = np.abs((y_cnnr[0][-1]-ex_sol)/ex_sol)

                    return error_ee, error_rk4, error_cnnr
                
                for i in range(len(deltastep)):
                    rerr_ee[i], rerr_rk4[i], rerr_cnnr[i] = cons(y0, int(deltastep[i]), tolerance)
                    deltastept[i] = tM/deltastep[i]
                
                ## PLOTS
                
                # ORDER OF CONVERGENCE ESTIMATION

                # Find indices where RK4 error is clean (above the noise floor)
                cleanrk4 = rerr_rk4 > 1e-13
                
                p_ee = np.polyfit(np.log(deltastept),np.log(rerr_ee),1)
                p_rk4clean = np.polyfit(np.log(deltastept[cleanrk4]),np.log(rerr_rk4[cleanrk4]),1)
                #p_cne = np.polyfit(np.log(deltastept),np.log(rerr_cne),1)
                #p_cnfp = np.polyfit(np.log(deltastept),np.log(rerr_cnfp),1)
                p_cnnr = np.polyfit(np.log(deltastept),np.log(rerr_cnnr),1)

                plt.figure(figsize=(12,9))
                plt.loglog(deltastept, rerr_ee, label = 'EE, p = ' + str(p_ee[0]))
                plt.loglog(deltastept, rerr_rk4, label = 'RK4, p = ' + str(p_rk4clean[0]))
                #plt.loglog(deltastept, rerr_cne, label = 'CN with EE initial guess, p = ' + str(p_cne[0]))
                #plt.loglog(deltastept, rerr_cnfp, label = 'CN with EE initial guess, fp iterations, p = ' + str(p_cnfp[0]))
                plt.loglog(deltastept, rerr_cnnr, label = 'CN with EE initial guess, nr iterations, p = ' + str(p_cnnr[0]))
                plt.legend(loc='best')
                plt.xlabel('Deltat (s) ', fontweight = 'bold')
                plt.ylabel('Relative error', fontweight = 'bold')
                plt.title('Evolution of relative error with different methods')
                plt.grid(True, which = 'both')
                plt.show()
