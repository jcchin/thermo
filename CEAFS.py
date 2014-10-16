import math
from numpy import *
import numpy as np


R =1.987 #universal gas constant


class CEAFS(object):    #trigger action on Mach
    
    a=np.array([
    [ #CO
    4.619197250e+05,-1.944704863e+03,5.916714180e+00,-5.664282830e-04,
    1.398814540e-07,-1.787680361e-11,9.620935570e-16,-2.466261084e+03,
    -1.387413108e+01
    ],
    [#CO2
    1.176962419e+05,-1.788791477e+03,8.291523190e+00,-9.223156780e-05,
    4.863676880e-09,-1.891053312e-12,6.330036590e-16,-3.908350590e+04,
    -2.652669281e+01
    ],
    [ #O2
    -1.037939022e+06,2.344830282e+03,1.819732036e+00,1.267847582e-03,
    -2.188067988e-07,2.053719572e-11,-8.193467050e-16,-1.689010929e+04,
    1.738716506e+01
    ]
    ])
    wt_mole = np.array([ 28.01, 44.01, 32. ])    
    element_wt = [ 12.01, 16.0 ]
    aij = np.array([ [1,1,0], [1,2,2] ])

    _num_element = 2
    _num_react = 3


    def __init__(self): 

        self._n = empty(self._num_react+1, dtype='complex')

        self._muj= zeros(self._num_react, dtype='complex')     

        self._Chem = np.zeros((self._num_element+1, self._num_element+1), dtype='complex') 
        self._Temp = np.zeros((self._num_element+1, self._num_element+1), dtype='complex') 
        self._Press= np.zeros((self._num_element+1, self._num_element+1), dtype='complex') 

        self._rhs = np.zeros(self._num_element + 1, dtype='complex')

        self._results = np.zeros(self._num_element + 1, dtype='complex')

        

        self._bsubi = np.zeros(self._num_element, dtype='complex')
        self._bsub0 = np.zeros(self._num_element, dtype='complex')

        self._dLn = np.zeros(self._num_react, dtype="complex")

        self.T = 0 #Kelvin
        self.P = 0 #Bar

        
    def H0( self, T ):
        ai = self.a.T
        return (-ai[0]/T**2 + ai[1]/T*np.log(T) + ai[2] + ai[3]*T/2 + ai[4]*T**2/3 + ai[5]*T**3/4 + ai[6]*T**4/5+ai[7]/T)
    
    def S0( self, T ):
        ai = self.a.T
        return (-ai[0]/(2*T**2) - ai[1]/T + ai[2]*np.log(T) + ai[3]*T + ai[4]*T**2/2 + ai[5]*T**3/3 + ai[6]*T**4/5+ai[8] )
    
    def Cp0( self, T):
        ai = self.a.T
        return ai[0]/T**2 + ai[1]/T + ai[2] + ai[3]*T + ai[4]*T**2 + ai[5]*T**3 + ai[6]*T**4 
    
    
    def _eq_init(self): 

        num_react = self._num_react
        num_element = self._num_element
        bsub0 = self._bsub0

        nj = self._n[:-1]
        muj = self._muj

        nj[:] = [0.,1.,0.] #initial reactant mixture assumes all CO2
        self._n[-1] = .1 #initial MW weight

        #deterine bsub0 - compute the distribution of elements from the distribution of reactants
        for i in range( 0, num_element ):
            sum_aij_nj = np.sum(self.aij[i] * nj / self.wt_mole)
            bsub0[ i ] = sum_aij_nj    
                               
        self._nmoles = .1 #CEA initial guess for a MW of 30 for final mixture
        nj[:] = ones(num_react, dtype='complex')/num_react #reset initial guess to equal concentrations


    def set_total_TP(self, T, P ):

        nj = self._n[:-1]
        num_element = self._num_element
        num_react = self._num_react
        results = self._results
        rhs = self._rhs

        self._eq_init()

        self.T = T
        self.P = P

        #Gauss-Seidel Iteration
        count = 0    
        while count < 20:
            count = count + 1
            R = self._resid_TP(self._n)
            self._n *= R

        sum_nj = np.sum(nj)


        #rhs for Cp constant p 
        rhs[:num_element] = self._bsubi
        rhs[num_element] = np.sum(nj)

        self._Press[num_element, num_element] = 0
        results = linalg.solve( self._Press, rhs )

        dlnVqdlnP = -1 + results[num_element]  

        #rhs for Cp constant T
        H0_T = self.H0(T) #compute this once, and use it a bunch of times

        #determinerhs 2.56
        for i in range( 0, num_element ):
            sum_aij_nj_Hj = np.sum(self.aij[i]*nj*H0_T)
            rhs[i]=sum_aij_nj_Hj


        #determinerhs 2.58
        sum_nj_Hj = np.sum(nj*H0_T)
        rhs[num_element]=sum_nj_Hj

        self._Temp[num_element, num_element] = 0
        results = linalg.solve( self._Temp, rhs )


        dlnVqdlnT = 1 - results[num_element]

        Cpf = np.sum(nj*self.Cp0(T))

        Cpe = 0 
        for i in range( 0, num_element ):
            sum_aijnjhj = np.sum(self.aij[i]*nj*H0_T)
            Cpe += sum_aijnjhj*results[i]
        Cpe += np.sum(nj**2*H0_T**2)
        Cpe += np.sum(nj*H0_T*results[num_element])

        self.Cp = (Cpe+Cpf)*1.987
        self.Cv = self.Cp + self._n[-1]*1.987*dlnVqdlnT**2/dlnVqdlnP
        self.gamma = -1*( self.Cp / self.Cv )/dlnVqdlnP

        return nj/sum_nj

    def _resid_TP(self, nj_guess): 

        num_react = self._num_react
        num_element = self._num_element

        chmatrix = self._Chem
        pmatrix = self._Temp
        tmatrix = self._Press

        rhs = self._rhs
        results = self._results
        nmoles = self._nmoles
        bsub0 = self._bsub0
        bsubi = self._bsubi

        dLn = self._dLn

        muj = self._muj
        n = nj_guess.copy()
        nj= n[:-1]
        nmoles = n[-1]
        
        #calculate mu for each reactant
        muj = self.H0(self.T) - self.S0(self.T) + np.log(nj) + np.log(self.P/nmoles) #pressure in Bars

        #calculate b_i for each element
        for i in range( 0, num_element ):
            bsubi[ i ] =  np.sum(self.aij[i]*nj) 

        ##determine pi coef for 2.24, 2.56, and 2.64 for each element
        for i in range( 0, num_element ):
            for j in range( 0, num_element ):
                tot = np.sum(self.aij[i]*self.aij[j]*nj)
                chmatrix[i][j] = tot
                tmatrix[i][j] = tot
                pmatrix[i][j] = tot


        #determine the delta n coeff for 2.24, dln/dlnT coeff for 2.56, and dln/dlP coeff 2.64
        #and pi coef for 2.26,  dpi/dlnT for 2.58, and dpi/dlnP for 2.66
        #and rhs of 2.64
        
        #determine the delta coeff for 2.24 and pi coef for 2.26\  
        chmatrix[num_element,:-1]=bsubi
        chmatrix[:-1,num_element]=bsubi
        tmatrix[num_element,:-1] =bsubi
        tmatrix[:-1,num_element] =bsubi
        pmatrix[num_element,:-1] =bsubi
        pmatrix[:-1,num_element] =bsubi

        #determine delta n coef for eq 2.26
        sum_nj = np.sum(nj)
        chmatrix[num_element][num_element] = sum_nj- nmoles

        #determine right side of matrix for eq 2.24
        for i in range( 0, num_element ):
            sum_aij_nj_muj = np.sum(self.aij[i]*nj*muj)
            rhs[i]=bsub0[i]-bsubi[i]+sum_aij_nj_muj

        #determine the right side of the matrix for eq 2.36
        sum_nj_muj = np.sum(nj*muj)
        rhs[num_element] = nmoles - sum_nj + sum_nj_muj

        #solve it
        results = linalg.solve( chmatrix, rhs )
        
        #determine lamdba eqns 3.1, 3.2, amd 3.3
        max = abs( 5*results[num_element] )
        
        for j in range( 0, num_react ):
            sum_aij_pi = np.sum(self.aij[:,j]*results[:-1])
            dLn[j] = results[num_element]+sum_aij_pi-muj[j]
            if abs( dLn[j] ) > max:
                max = abs( dLn[j] )
        
        lambdaf = 2 / (max+1e-20)
        if ( lambdaf > 1 ):
            lambdaf = 1 

        #update total moles eq 3.4
        new_nmoles = exp(lambdaf*results[-1])
        #update each reactant moles eq 3.4 and 2.18
        for j in range( 0, num_react ):
            sum_aij_pi = np.sum(self.aij[:,j]*results[:-1])
            dLn[j] = results[num_element]+sum_aij_pi-muj[j]
            nj[j] = exp(lambdaf*dLn[j])

        #return nj/np.sum(nj)
        print nj, new_nmoles
        #return np.hstack((nj-nj_guess[:-1], new_nmoles-nmoles))
        return np.hstack((nj,new_moles))
        #return nj

                


               
                
                
                
            


