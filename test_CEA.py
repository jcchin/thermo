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


if __name__ == "__main__": 
    unittest.main()




