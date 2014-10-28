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

    def test_1500K(self): 

        self.cea.set_total_TP( 1500, 1.034210 ) #kelvin, bars

        relative_concentrations = self.cea._n[:-1].real/np.sum(self.cea._n[:-1].real)
        goal = np.array([0.00036, 0.99946, 0.00018])
        error = relative_concentrations - goal 
        self.assertTrue(np.all(error < 1e-4))

        self.assertAlmostEqual(self.cea.Cp.real, 0.32649940109638081, 2)
        self.assertAlmostEqual(self.cea.gamma.real, 1.1614472804210347, 3)

class Deriv_Tests(unittest.TestCase):
    def setUp(self): 
        self.cea = CEAFS(); 

    def test_pi2n_applyJ(self):

        base_pi = [ -1.77208405e+01,  -1.63009204e+01,  -1.34226676e-09,] 
        base_muj =[-34.02175706, -50.32264785, -32.60178159,]


        base_n = self.cea._pi2n(base_pi,base_muj)

        #print "basen", base_n

        for i in xrange (len(base_pi)):
            # inaccurate fd
            # delta = list(base_pi)
            # delta[i] *= 1.0001
            # new_n = (self.cea._pi2n(delta,base_muj)-base_n).real
            # fd = (new_n/(delta[i]-base_pi[i]))

            delta = list(base_pi)
            delta[i] += .01j
            cs = (self.cea._pi2n(delta,base_muj)-base_n).imag/.01

            vec_pi = np.zeros(len(base_pi))
            vec_pi[i] = 1
            vec_muj = np.zeros(len(base_muj))

            analytic = self.cea._pi2n_applyJ(vec_pi,vec_muj).real
            error = abs(analytic-cs)
            self.assertTrue(np.all(error < 1e-5))

    def test_n2pi_applyL(self):

        base_n_guess = [ 0.02040748+0.j,  0.00231478+0.j,  0.01020431+0.j,  0.03292581+0.j]
        
        base_chmatrix, base_rhs, base_muj = self.cea._n2pi(base_n_guess)

        delta_n_guess = np.copy(base_n_guess)
        delta_n_guess[0] += .01j

        pert_chmatrix, pert_rhs, pert_muj = self.cea._n2pi(delta_n_guess)

        #print (pert_chmatrix[0][0] - base_chmatrix[0][0]).imag/.01
        print 30*"-"
        print base_muj
        print pert_muj
        print "!!!", (pert_muj - base_muj).imag/.01
        self.assertTrue(True)



if __name__ == "__main__": 
    unittest.main()




