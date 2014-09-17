


import math
from numpy import *
import numpy as np
class CEAFS():    #trigger action on Mach
    
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
    wt_mole = [ 28.01, 44.01, 32. ]    
    element_wt = [ 12.01, 16.0 ]
    aij = np.array([ [1,1,0], [1,2,2] ])
    nj = [ 1, 0, 0  ]

        
    def H0( self, T, species ):
        R =1.987
        return (-self.a[species][0]/T**2 + self.a[species][1]/T*math.log(T) + self.a[species][2] + self.a[species][3]*T/2 + self.a[species][4]*T**2/3 + self.a[species][5]*T**3/4 + self.a[species][6]*T**4/5+self.a[species][7]/T)
    
    def S0( self, T, species ):
        R =1.987
        return (-self.a[species][0]/(2*T**2) - self.a[species][1]/T + self.a[species][2]*math.log(T) + self.a[species][3]*T + self.a[species][4]*T**2/2 + self.a[species][5]*T**3/3 + self.a[species][6]*T**4/5+self.a[species][8] )
    
    def Cp( self, T, species ):
        return self.a[species][0]/T**2 + self.a[species][1]/T + self.a[species][2] + self.a[species][3]*T + self.a[species][4]*T**2 + self.a[species][5]*T**3 + self.a[species][6]*T**4 
    
    def matrix(self,T, P ):

        num_element = 2
        num_react = 3
        sum_njhjh = 0
        sum_njmuj = 0
        sum_mujnjhj = 0
        #sum_aij_nj =[][]
        bsubi = np.zeros(num_element)
        bsub0 = np.zeros(num_element)
                
        #deterine bsub0 - compute the distribution of elements from the distribution of reactants
        for i in range( 0, num_element ):
            sum_aij_nj = 0
            nj=[.0,1.,.0] #nj is the initial reactant mixture
            for j in range( 0, num_react ):
                sum_aij_nj +=  ( self.aij[i][j]*nj[j] )/self.wt_mole[j]
            bsub0[ i ] = sum_aij_nj    
                        
        print bsub0
        
        nmoles = .1 #CEA initial guess for a MW of 30 for final mixture
        nj = ones(num_react)/num_react
        muj= zeros(num_react)           
        sj= [[0 for x in xrange(5)] for x in xrange(5)] 

        sum_nj_hj = 0.
        sum_nj_hj_h = 0.
        sum_nj_muj = 0.
        sum_muj_nj_hj = 0.
        sum_aij_nj= [[0 for x in xrange(5)] for x in xrange(5)] 

        chmatrix= [[0 for x in xrange(3)] for x in xrange(3)]
        b = np.zeros(num_element + 1)
        dLn = np.zeros(num_react)
        count = 0
            
        
        while count< 20:
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
                    sum = 0
                    for k in range( 0, num_react ):
                        sum = sum +  self.aij[i][k]*self.aij[j][k]*nj[k]
                    chmatrix[i][j] = sum
            
            #determine the delta coeff for 2.24 and pi coef for 2.26
            for i in range( 0, num_element ):
                chmatrix[num_element][i]=bsubi[i]
                chmatrix[i][num_element]=bsubi[i]
        
            #determine delta n coef for eq 2.25
            sum_nj = 0 
            for i in range( 0, num_react ):
                sum_nj = sum_nj + nj[i]
            chmatrix[num_element][num_element]=sum_nj-nmoles
 
            #determine right side of matrix for eq 2.24
            for i in range( 0, num_element ):
                sum_aij_nj_muj = np.sum(self.aij[i]*nj*muj)
                b[i]=bsub0[i]-bsubi[i]+sum_aij_nj_muj

            #determine the right side of the matrix for eq 2.25
            sum_nj_muj = 0
            for j in range( 0, num_react ):
                sum_nj_muj = sum_nj_muj + nj[j]*muj[j]
            b[num_element]=nmoles - sum_nj +sum_nj_muj

            #print b
            #solve it
            results = linalg.solve( chmatrix, b )
            #print chmatrix,b
            
            #print results
            #determine lamdba eqns 3.1, 3.2, amd 3.3
            max = 0
            if abs( 5*results[num_element] ) > max: 
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
            #print 'll',lambdaf
            #update total moles eq 3.4
            nmoles = exp( math.log( nmoles ) + lambdaf*results[num_element] )

            #update each reactant moles eq 3.4 and 2.18
            for j in range( 0, num_react ):
                sum_aij_pi = 0
                for i in range( 0, num_element ):
                    sum_aij_pi = sum_aij_pi+self.aij[i][j] * results[i]
                dLn[j] = results[num_element]+sum_aij_pi-muj[j]
                nj[j]= exp( math.log( nj[j] ) + lambdaf*dLn[j] )

            

            print array(nj)/np.sum(nj)
            print
            print

                


               
                
                
                
            


