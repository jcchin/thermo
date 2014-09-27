import numpy as np
from matplotlib import pyplot as plt

from CEAFS import CEAFS

CEA = CEAFS();

P_base = 1.03
T_base = 4000

baseline = CEA.set_total_TP( T_base, P_base ) #kelvin, bars

print CEA._nj.real/np.sum(CEA._nj.real)
print CEA.Cp.real
print CEA.gamma.real

fd_temp = []
cs_temp= []
steps_temp = []

#temperature derivatives
for p in np.linspace(1,15,50): 
    step = 10.0**(-1*p)
    steps_temp.append(step)

    fd_step = CEA.set_total_TP(T_base*(1+step), P_base).real
    cplex_step = CEA.set_total_TP(complex(T_base,step), P_base)

    fd_temp.append((fd_step-baseline)/(T_base*step))
    cs_temp.append(cplex_step.imag/step)

    # print "step: %1.0e,     fd: %s"%(step,(fd_step-baseline).real/(step))
    # print "             c-step:%s"% (cplex_step.imag/step)
    # print 

fd_temp = np.array(fd_temp)
cs_temp = np.array(cs_temp)

#pressure derivatives
fd_press = []
cs_press= []
steps_press = []

for p in np.linspace(1,15,50): 
    step = 10.0**(-1*p)
    steps_press.append(step)

    fd_step = CEA.set_total_TP(T_base, P_base*(1+step))
    cplex_step = CEA.set_total_TP(T_base, complex(P_base,step))

    fd_press.append((fd_step-baseline).real/(P_base*step))
    cs_press.append(cplex_step.imag/step)

    # print "step: %1.0e,     fd: %s"%(step,(fd_step-baseline).real/(1.03*step))
    # print "             c-step:%s"% (cplex_step.imag/step)
    # print 

fd_press = np.array(fd_press)
cs_press = np.array(cs_press)

#Plotting stuff
mag = np.max(fd_temp[:,0])
# plt.semilogx(steps_temp, fd_temp[:,0]/mag, c='b', label=r"FD: dCO_dT")
# plt.semilogx(steps_temp, cs_temp[:,0]/mag, c='b', ls="--", lw=2.5, label=r"CS: dCO_dT")
# plt.legend(loc="best")
plt.loglog(steps_temp, abs(fd_temp[:,0]-cs_temp[:,0]), c='b', label="Temperature")


#plt.figure()
mag = np.min(fd_press[:,0])
#plt.semilogx(steps_press, fd_press[:,0]/mag, c='g', label=r"FD: dCO_dP")
#plt.semilogx(steps_press, cs_press[:,0]/mag, c='g', ls="--", lw=2.5, label=r"CS: dCO_dP")


plt.loglog(steps_press, abs(fd_press[:,0]-cs_press[:,0]), c='g', label="Pressure")
plt.ylabel('log(abs(Error))')
plt.xlabel('log(relative step size)')
plt.title('Error in FD Derivatives vs Step Size')
plt.legend(loc="best")
plt.show()



