


import math
from numpy import *
import numpy as np


R =1.987 #universal gas constant


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
        return (-self.a[species][0]/T**2 + self.a[species][1]/T*np.log(T) + self.a[species][2] + self.a[species][3]*T/2 + self.a[species][4]*T**2/3 + self.a[species][5]*T**3/4 + self.a[species][6]*T**4/5+self.a[species][7]/T)
    
    def S0( self, T, species ):
        return (-self.a[species][0]/(2*T**2) - self.a[species][1]/T + self.a[species][2]*np.log(T) + self.a[species][3]*T + self.a[species][4]*T**2/2 + self.a[species][5]*T**3/3 + self.a[species][6]*T**4/5+self.a[species][8] )
    
    def Cp( self, T, species ):
        return self.a[species][0]/T**2 + self.a[species][1]/T + self.a[species][2] + self.a[species][3]*T + self.a[species][4]*T**2 + self.a[species][5]*T**3 + self.a[species][6]*T**4 
    
    def matrix(self,T, P ):

        num_element = 2
        num_react = 3
        sum_njhjh = 0
        sum_njmuj = 0
        sum_mujnjhj = 0
        #sum_aij_nj =[][]
        bsubi = np.zeros(num_element, dtype='complex')
        bsub0 = np.zeros(num_element, dtype='complex')
                
        #deterine bsub0 - compute the distribution of elements from the distribution of reactants
        for i in range( 0, num_element ):
            sum_aij_nj = 0
            nj=[.0,1.,.0] #nj is the initial reactant mixture
            for j in range( 0, num_react ):
                sum_aij_nj +=  ( self.aij[i][j]*nj[j] )/self.wt_mole[j]
            bsub0[ i ] = sum_aij_nj    
                               
        nmoles = .1 #CEA initial guess for a MW of 30 for final mixture
        nj = ones(num_react, dtype='complex')/num_react
        muj= zeros(num_react, dtype='complex')           

        chmatrix= np.zeros((num_element+1, num_element+1), dtype='complex')
        pmatrix= np.zeros((num_element+1, num_element+1), dtype='complex')
        tmatrix= np.zeros((num_element+1, num_element+1), dtype='complex')
        rhs = np.zeros(num_element + 1, dtype='complex')
        rhstmatrix = np.zeros(num_element + 1, dtype='complex')
        rhspmatrix = np.zeros(num_element + 1, dtype='complex')        
        results = np.zeros(num_element + 1, dtype='complex')
        results_old = np.zeros(num_element + 1, dtype='complex')
        dLn = np.zeros(num_react, dtype="complex")


        
        count = 0    
        
        while count< 20:
            count = count + 1
                
            #calculate mu for each reactant
            for i in range( 0, num_react ):
                muj[i] = self.H0( T, i ) - self.S0(T,i) + np.log( nj[i] ) + np.log( P / nmoles ) #pressure in Bars

            #calculate b_i for each element
            for i in range( 0, num_element ):
                bsubi[ i ] =  np.sum(self.aij[i]*nj) 
   
            #determine pi coef for 2.24, 2.56, and 2.64 for each element
            for i in range( 0, num_element ):
                for j in range( 0, num_element ):
                    tot = 0
                    for k in range( 0, num_react ):
                        tot += self.aij[i][k]*self.aij[j][k]*nj[k]
                    chmatrix[i][j] = tot
                    pmatrix[i][j]=tot
                    tmatrix[i][j]=tot
                    
            
            #determine the delta n coeff for 2.24, dln/dlnT coeff for 2.56, and dln/dlP coeff 2.64
            #and pi coef for 2.26,  dpi/dlnT for 2.58, and dpi/dlnP for 2.66
            #and rhs of 2.64
            for i in range( 0, num_element ):
                chmatrix[num_element][i]=bsubi[i]
                chmatrix[i][num_element]=bsubi[i]
                tmatrix[num_element][i]=bsubi[i]
                tmatrix[i][num_element]=bsubi[i]
                pmatrix[num_element][i]=bsubi[i]
                pmatrix[i][num_element]=bsubi[i]                

                
            for i in range( 0, num_element ):
                rhspmatrix[i]=bsubi[i]
                    
        
            #determine delta n coef for eq 2.26
            sum_nj = np.sum(nj)
            chmatrix[num_element][num_element] = sum_nj- nmoles
            #determine rhs of eq 2.66
            rhspmatrix[num_element]= sum_nj
            
            #set the dln/dlnT coeff for 2.58
            pmatrix[num_element][num_element]=0

            #set the dln/dlnT coeff for 2.66
            pmatrix[num_element][num_element]=0
            
            #determine right side of matrix for eq 2.24
            for i in range( 0, num_element ):
                sum_aij_nj_muj = np.sum(self.aij[i]*nj*muj)
                rhs[i]=bsub0[i]-bsubi[i]+sum_aij_nj_muj

            #determine the right side of the matrix for eq 2.36
            sum_nj_muj = np.sum(nj*muj)
            rhs[num_element] = nmoles - sum_nj + sum_nj_muj
                                
            #determinerhs 2.56
            for i in range( 0, num_element ):
                sum_aij_nj_Hj = 0    
                for j in range( 0, num_react ):
                    sum_aij_nj_Hj = sum_aij_nj_Hj+self.aij[i][j]*nj[j]*self.H0(T,j)
            rhstmatrix[i]=sum_aij_nj_Hj

                      
            #determinerhs 2.58
            sum_nj_Hj = 0
            for j in range( 0, num_react ):
                sum_nj_Hj += nj[j]*self.H0(T,j)
            rhstmatrix[num_element]=sum_nj_Hj
        
            #solve it
            results_old = results.copy()
            results = linalg.solve( chmatrix, rhs )
           
            resultsp = linalg.solve( pmatrix, rhspmatrix )
            resultst = linalg.solve( tmatrix, rhstmatrix )
            

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

        h = 0.
        for j in range( 0, num_react ):
            h = h + nj[j]*self.H0( T, j )*8314.51/4184.*T

        s = 0
        for j in range( 0, num_react ):
            print nj[j]/nmoles
            s = s + nj[j]*(self.S0( T, j )*8314.51/4184. -8314.51/4184.*log( nj[j]/nmoles)-8314.51/4184.*log( P ))
                     
        Cpf = 0
        for j in range( 0, num_react ):
            Cpf = Cpf + nj[j]*self.Cp( T, j )
                           
        dlnVqdlnT = 1 + resultsp[num_element]
        dlnVqdlnP = -1 + resultst[num_element]    
            
        Cpe = 0    
        for i in range( 0, num_element ):
            sum_aijnjhj = 0
            Cpr = 0
            for j in range( 0, num_react ):
                sum_aijnjhj = sum_aijnjhj + self.aij[i][j]*nj[j]*self.H0(T,j)
            Cpe = Cpe +  sum_aijnjhj*resultst[i]

        for j in range( 0, num_react ):
            Cpe = Cpe + nj[j]**2*self.H0(T,j)**2;
            
        for j in range( 0, num_react ):
            Cpe = Cpe + nj[j]*self.H0(T,j)*resultst[num_element];

        rho = P*100/(nmoles*8314*T)            
        dlnVqdlnT = 1 - resultst[num_element]
        dlnVqdlnP = -1 + resultsp[num_element]    
        
        Cp = (Cpe+Cpf)*1.987

        Cv = Cp + nmoles*1.987*dlnVqdlnT**2/dlnVqdlnP

        gamma = -1*( Cp / Cv )/dlnVqdlnP

        print "HERE"
        print gamma,Cp
        print nj
        return nj

                


               
                
                
                
            


