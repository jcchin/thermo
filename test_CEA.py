import unittest

import numpy as np

from CEAFS import CEAFS


class CEA_TestCase(unittest.TestCase): 

    def setUp(self): 
        self.cea = CEAFS(); 


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
        self.cea = CEAFS(); 

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


    def test_n2pi_applyJ(self): 

        self.cea.set_total_TP( 1500, 1.034210 ) #kelvin, bars 
        base_n = np.array([7.94249751e-06, 2.27142886e-02, 4.29623938e-06, 2.27260187e-02], dtype='complex')
        base_chmatrix, base_pi, base_muj = self.cea._n2pi(base_n)

        #dmuj_dn
        for i in xrange(base_n.shape[0]): 
            
            delta = base_n.copy()
            delta[i] += complex(0,1e-40) #derivative starts to explode for ln(small values), so need a really small step to get accuracy
            new_chmatrix, new_pi, new_muj = self.cea._n2pi(delta)
            cs_muj = new_muj.imag/1e-40
            cs_pi = new_pi.imag/1e-40
            cs_chmatrix = new_chmatrix.imag/1e-40

            # delta = base_n.copy()
            # delta[i] *= 1.001
            # new_chmatrix, new_pi, new_muj = self.cea._n2pi(delta)
            # fd_muj = (new_muj-base_muj).real/(base_n[i]*.001)

            vec_n = np.zeros(base_n.shape)
            vec_n[i] = 1
            a_chmatrix, a_pi, a_muj = self.cea._n2pi_applyJ(vec_n)

            error = np.abs(a_muj.real-cs_muj)
            self.assertTrue(np.all(error < 1e-3))

            error = np.abs(a_pi.real-cs_pi)
            self.assertTrue(np.all(error < 1e-3))

            error = np.abs(a_chmatrix.real-cs_chmatrix)
            self.assertTrue(np.all(error < 1e-3))





if __name__ == "__main__": 
    unittest.main()




