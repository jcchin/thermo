import unittest

import numpy as np

from CEAFS import CEAFS


class CEA_TestCase(unittest.TestCase): 

    def setUp(self): 
        self.cea = CEAFS(dtype="complex"); 


    def test_4000K(self): 

        self.cea.set_total_TP( 4000, 1.034210 ) #kelvin, bars

        relative_concentrations = self.cea._n[:-1].real/np.sum(self.cea._n[:-1].real)
        goal = np.array([0.61976,0.07037,0.30988])
        error = relative_concentrations - goal 
        self.assertTrue(np.all(error < 1e-4))

        self.assertAlmostEqual(self.cea.Cp.real, 0.55868412681771595, 2)
        self.assertAlmostEqual(self.cea.gamma.real, 1.1997550763532066, 3)
        self.assertAlmostEqual(self.cea.h.real, 340.3269, 2)
        self.assertAlmostEqual(self.cea.s.real, 2.3576, 3)
        self.assertAlmostEqual(self.cea.rho.real, 9.4447e-5, 3)

    def test_1500K(self): 

        self.cea.set_total_TP( 1500, 1.034210 ) #kelvin, bars

        relative_concentrations = self.cea._n[:-1].real/np.sum(self.cea._n[:-1].real)
        goal = np.array([0.00036, 0.99946, 0.00018])
        error = relative_concentrations - goal 
        self.assertTrue(np.all(error < 1e-4))

        self.assertAlmostEqual(self.cea.Cp.real, 0.32649940109638081, 2)
        self.assertAlmostEqual(self.cea.gamma.real, 1.1614472804210347, 3)
        self.assertAlmostEqual(self.cea.h.real, -1801.3894, 2)
        self.assertAlmostEqual(self.cea.s.real, 1.5857, 3)
        self.assertAlmostEqual(self.cea.rho.real, 3.6488e-4, 3)
        
class Deriv_Tests(unittest.TestCase):
    def setUp(self): 
        self.cea = CEAFS(dtype="complex"); 

    def test_pi2n_applyJ(self):

        base_pi = [ -1.77208405e+01,  -1.63009204e+01,  -1.34226676e-09,] 
        base_muj =[-34.02175706, -50.32264785, -32.60178159,]


        base_n = self.cea._pi2n(base_pi,base_muj)

        #check pi inputs
        for i in xrange (len(base_pi)):
            # inaccurate fd
            # delta = list(base_pi)
            # delta[i] *= 1.0001
            # new_n = (self.cea._pi2n(delta,base_muj)-base_n).real
            # fd = (new_n/(delta[i]-base_pi[i]))

            delta = list(base_pi)
            delta[i] += .01j
            cs = self.cea._pi2n(delta,base_muj).imag/.01

            vec_pi = np.zeros(len(base_pi))
            vec_pi[i] = 1
            vec_muj = np.zeros(len(base_muj))

            analytic = self.cea._pi2n_applyJ(vec_pi,vec_muj).real
            error = np.abs(analytic-cs)
            self.assertTrue(np.all(error < 1e-5))

        #check muj input
        for i in xrange (len(base_muj)):

            delta = list(base_muj)
            delta[i] += .01j
            cs = self.cea._pi2n(base_pi,delta).imag/.01

            vec_muj = np.zeros(len(base_muj))
            vec_muj[i] = 1
            vec_pi = np.zeros(len(base_pi))

            analytic = self.cea._pi2n_applyJ(vec_pi,vec_muj).real
            error = np.abs(analytic-cs)
            self.assertTrue(np.all(error < 1e-5))

    def test_H0_S0_Cp_applyJ(self): 

        base_T = 1500
        self.cea.T = base_T
        base_H0 = self.cea.H0()
        base_S0 = self.cea.S0()
        base_Cp0 = self.cea.Cp0()

        step_size = 1e-40
        delta_T = complex(self.cea.T,step_size)
        
        self.cea.T = delta_T
        new_H0 = self.cea.H0()
        dH0_cs = new_H0.imag/step_size

        new_S0 = self.cea.S0()
        dS0_cs = new_S0.imag/step_size

        new_Cp0 = self.cea.Cp0()
        dCp0_cs = new_Cp0.imag/step_size


        dH0_a = self.cea._H0_applyJ(1.)
        error = dH0_a - dH0_cs
        self.assertTrue(np.linalg.norm(error) < 1e-5)

        dS0_a = self.cea._S0_applyJ(1.)
        error = dS0_a - dS0_cs
        self.assertTrue(np.linalg.norm(error) < 1e-5)

        dCp0_a = self.cea._Cp0_applyJ(1.)
        error = dCp0_a - dCp0_cs
        self.assertTrue(np.linalg.norm(error) < 1e-5)




    def test_n2ls_applyJ(self): 

        self.cea.T = 1500 
        self.cea.P = 1.034210
        self.cea.set_total_TP( 1500, 1.034210 ) #kelvin, bars
        base_n = self.cea._n.copy()
        base_chmatrix, base_rhs, base_muj = self.cea._n2ls(base_n)

        #copy because cea element used a single rhs vector, which gets changes. 
        base_chmatrix = base_chmatrix.copy()
        base_rhs = base_rhs.copy()
        base_muj = base_muj.copy()

        for i in xrange(base_n.shape[0]): 
            
            delta = base_n.copy()
            step = 1e-15
            delta[i] += complex(0,step) #derivative starts to explode for ln(small values), so need a really small step to get accuracy
            new_chmatrix, new_rhs, new_muj = self.cea._n2ls(delta)
            cs_muj = new_muj.imag/step
            cs_rhs = new_rhs.imag/step
            cs_chmatrix = new_chmatrix.imag/step

            # delta = base_n.copy()
            # delta[i] *= 1.000001
            # new_chmatrix, new_rhs, new_muj = self.cea._n2ls(delta)
            # fd_muj = (new_muj-base_muj).real/(base_n[i].real*.000001)
            # fd_rhs = (new_rhs-base_rhs).real/(base_n[i].real*.000001)


            #reset the 
            vec_n = np.zeros(base_n.shape)
            vec_n[i] = 1
            a_chmatrix, a_rhs, a_muj = self.cea._n2ls_applyJ(vec_n, 0,0)
            a_chmatrix = a_chmatrix.real
            a_rhs = a_rhs.real
            a_muj = a_muj.real 

            error = np.abs((a_muj.real-cs_muj)/(cs_muj+1e-50))
            self.assertTrue(np.all(error < 1e-5))

            #using relative error here
            error = np.abs((a_rhs.real-cs_rhs)/(cs_rhs+1e-90)) #1e-50 protects against divide by zero errors
            self.assertTrue(np.all(error < 1e-5))

            # error = np.abs(a_chmatrix.real-cs_chmatrix)
            # self.assertTrue(np.all(error < 1e-3))
        

        #------- T
        blank_n = np.zeros(base_n.shape)
        self.cea.T = complex(1500,1e-40) #this is set on line 130

        new_chmatrix, new_rhs, new_muj = self.cea._n2ls(base_n)
        cs_muj = new_muj.imag/1e-40
        cs_rhs = new_rhs.imag/1e-40
        cs_chmatrix = new_chmatrix.imag/1e-40

        a_chmatrix, a_rhs, a_muj = self.cea._n2ls_applyJ(blank_n, 1,0)

        error = np.abs(a_muj.real-cs_muj)
        self.assertTrue(np.all(error < 1e-5))

        error = np.abs((a_rhs.real-cs_rhs)/(cs_rhs+1e-90))
        self.assertTrue(np.all(error < 1e-5))
        

        #------- P
        self.cea.P = complex(1.034210,1e-40) #this is set on line 130
        self.cea.T = 1500

        new_chmatrix, new_rhs, new_muj = self.cea._n2ls(base_n)
        cs_muj = new_muj.imag/1e-40
        cs_rhs = new_rhs.imag/1e-40
        cs_chmatrix = new_chmatrix.imag/1e-40

        a_chmatrix, a_rhs, a_muj = self.cea._n2ls_applyJ(blank_n, 0,1)

        error = np.abs((a_muj.real-cs_muj)/(cs_muj+1e-90))
        self.assertTrue(np.all(error < 1e-5))

        error = np.abs((a_rhs.real-cs_rhs)/(cs_rhs+1e-90))
        self.assertTrue(np.all(error < 1e-5))
        #self.assertTrue(np.all(error[-1] < 1e-3))
        




if __name__ == "__main__": 
    unittest.main()




