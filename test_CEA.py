import numpy as np

from CEAFS import CEAFS

CEA = CEAFS();

base = CEA.matrix( 4000, 1.0207 ) #kelvin, bars

# mole_frac = CEA.matrix(4000+.01j, 1.03)

# print "complex deriv:" 
# print mole_frac.imag/.01


# mole_frac = CEA.matrix(4000+.01j, 1.03)


# print 
# print "fd derivatives"
# for step in np.linspace(4001,4050,10): 
#     new = CEA.matrix(step, 1.03)

#     print (new-base).real/(4000*step - 4000)

#new = CEA.matrix (4001, 1.03).real
#new2 = CEA.matrix (4050, 1.03).real


#print "fd derivative:"
#print (new-base).real/(4001-4000)