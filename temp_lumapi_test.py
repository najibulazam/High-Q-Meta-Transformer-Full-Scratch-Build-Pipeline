import sys
sys.path.append(r'C:\Program Files\Lumerical\v241\api\python')
import lumapi
fd=lumapi.FDTD()
fd.eval('addcylinder; set("name", "Ge_Resonator");')
print('added', fd.getnamed('Ge_Resonator','name'))
