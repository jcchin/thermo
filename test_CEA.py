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

        self.assertAlmostEqual(self.cea.Cp.real, 0.56716387838, 4)
        self.assertAlmostEqual(self.cea.gamma.real, 1.19574894183, 4)


if __name__ == "__main__": 
    unittest.main()




