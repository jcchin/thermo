


import math
from numpy import *
import numpy as np


R =1.987 #universal gas constant


class CEAFS(object):    #trigger action on Mach
    
    a=[
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
    ]
    wt_mole = np.array([ 28.01, 44.01, 32. ])    
    element_wt = [ 12.01, 16.0 ]
    aij = np.array([ [1,1,0], [1,2,2] ])

    num_element = 2
    num_react = 3


    def __init__(self): 

    	self.nj = empty(self.num_react, dtype='complex')

        self.muj= zeros(self.num_react, dtype='complex')     

        chmatrix= np.zeros((num_element+1, num_element+1), dtype='complex')  
        
    def H0( self, T, species ):
        return (-self.a[species][0]/T**2 + self.a[species][1]/T*np.log(T) + self.a[species][2] + self.a[species][3]*T/2 + self.a[species][4]*T**2/3 + self.a[species][5]*T**3/4 + self.a[species][6]*T**4/5+self.a[species][7]/T)
    
    def S0( self, T, species ):
        return (-self.a[species][0]/(2*T**2) - self.a[species][1]/T + self.a[species][2]*np.log(T) + self.a[species][3]*T + self.a[species][4]*T**2/2 + self.a[species][5]*T**3/3 + self.a[species][6]*T**4/5+self.a[species][8] )
    
    def Cp( self, T, species ):
        return self.a[species][0]/T**2 + self.a[species][1]/T + self.a[species][2] + self.a[species][3]*T + self.a[species][4]*T**2 + self.a[species][5]*T**3 + self.a[species][6]*T**4 
    
    def matrix(self,T, P ):

        num_react = self.num_react
        num_element = self.num_element

        nj = self.nj
        muj = self.muj

        sum_njhjh = 0
        sum_njmuj = 0
        sum_mujnjhj = 0

        bsubi = np.zeros(num_element, dtype='complex')
        bsub0 = np.zeros(num_element, dtype='complex')

        nj[:] = [0.,1.,0.] #initial reactant mixture assumes all CO2

        #deterine bsub0 - compute the distribution of elements from the distribution of reactants
        for i in range( 0, num_element ):
            sum_aij_nj = 0
            sum_aij_nj = np.sum(self.aij[i] * nj / self.wt_mole)
            bsub0[ i ] = sum_aij_nj    
                               
        nmoles = .1 #CEA initial guess for a MW of 30 for final mixture
        
        nj = ones(num_react, dtype='complex')/num_react

        
        rhs = np.zeros(num_element + 1, dtype='complex')
        results = np.zeros(num_element + 1, dtype='complex')
        results_old = np.zeros(num_element + 1, dtype='complex')
        dLn = np.zeros(num_react, dtype="complex")

        count = 0    
        
        while count< 9:
            count = count + 1
                
            #calculate mu for each reactant
            for i in range( 0, num_react ):
                muj[i] = self.H0( T, i ) - self.S0(T,i) + np.log( nj[i] ) + np.log( P / nmoles ) #pressure in Bars

            #calculate b_i for each element
            for i in range( 0, num_element ):
                bsubi[ i ] =  np.sum(self.aij[i]*nj) 
   
            #determine pi coef for 2.24 for each element
            for i in range( 0, num_element ):
                for j in range( 0, num_element ):
                    tot = 0
                    for k in range( 0, num_react ):
                        tot += self.aij[i][k]*self.aij[j][k]*nj[k]
                    chmatrix[i][j] = tot
            
            #determine the delta coeff for 2.24 and pi coef for 2.26
            for i in range( 0, num_element ):
                chmatrix[num_element][i]=bsubi[i]
                chmatrix[i][num_element]=bsubi[i]
        
            #determine delta n coef for eq 2.26
            sum_nj = np.sum(nj)
            chmatrix[num_element][num_element] = sum_nj- nmoles
 
            #determine right side of matrix for eq 2.24
            for i in range( 0, num_element ):
                sum_aij_nj_muj = np.sum(self.aij[i]*nj*muj)
                rhs[i]=bsub0[i]-bsubi[i]+sum_aij_nj_muj

            #determine the right side of the matrix for eq 2.26
            sum_nj_muj = np.sum(nj*muj)
            rhs[num_element] = nmoles - sum_nj + sum_nj_muj

            #solve it
            results_old = results.copy()
            results = linalg.solve( chmatrix, rhs )
            
            #determine lamdba eqns 3.1, 3.2, amd 3.3
            max = abs( 5*results[num_element] )
            
            for j in range( 0, num_react ):
                sum_aij_pi = 0
                for i in range( 0, num_element ):
                    sum_aij_pi = sum_aij_pi+self.aij[i][j] * results[i]
                dLn[j] = results[num_element]+sum_aij_pi-muj[j]
                if abs( dLn[j] ) > max:
                    max = abs( dLn[j] )
            
            lambdaf = 2 / max
            if ( lambdaf > 1 ):
                lambdaf = 1 
            #update total moles eq 3.4
            nmoles = exp( np.log( nmoles ) + lambdaf*results[num_element] )

            #update each reactant moles eq 3.4 and 2.18
            for j in range( 0, num_react ):
                sum_aij_pi = 0
                for i in range( 0, num_element ):
                    sum_aij_pi = sum_aij_pi+self.aij[i][j] * results[i]
                dLn[j] = results[num_element]+sum_aij_pi-muj[j]
                nj[j]= exp( np.log( nj[j] ) + lambdaf*dLn[j] )

            
            #print count, np.linalg.norm(results - results_old)
            # print array(nj)/np.sum(nj)
            # print
            # print
		
		#print count, np.linalg.norm(results - results_old)
        #exit()
       	return nj/np.sum(nj)
       	#return nj

                


               
                
                
                
            


