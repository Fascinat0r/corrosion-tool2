import sys
import math

sys.path.insert(0,'../pycThermopack/')
from thermopack.cubic import cubic
from thermopack.saftvrmie import saftvrmie

def calculate_xtt (rho1, rho2, mu1, mu2, v1, D): #regime part
    return ((1.096/rho1)**0.5)*((rho1/rho2)**0.25)*((mu2/mu1)**0.1)*((v1/D)**0.5)

def calculate_visc (mu1, mu2, ff): #regime/friction part
    return ff*mu1+(1-ff)*mu2

def calculate_Re (V,D,rho, mu):#friction part
    return V*D*rho/mu

def return_lambda(Re):#friction part
    if (Re<2300): return 64/Re
    else: return 0.316/(Re**0.25)

def return_pressure_loss(V,D,L, lam, rho):#friction part
    xi = lam*L/D
    return (xi*V**2)*0.5*rho


def return_regime (xtt):#regime part
        if(xtt<10): return 'bubble'
        if(xtt>=10 & xtt<100): return 'plug'
        if(xtt>=100 & xtt<1000): return 'slug'
        if(xtt >= 1000 & xtt<10000): return 'annular'
        if (xtt>=10000): return 'mist'
        return 'undefined'
    
#liquid to solid viscosity calculation:

def return_friction_factor (xtt): #regime/friction part
        if(xtt<10): return 1
        if(xtt>=10 & xtt<100): return 0.9
        if(xtt>=100 & xtt<1000): return 0.8
        if(xtt >= 1000 & xtt<10000): return 0.7
        if (xtt>=10000): return 0.6
        return 0
#PVT block    
def eth_viscosity_from_temp (T):
    A = 0.00201*1e-6
    B = 1614
    C = 0.00618
    D = 1.132*(-1e-5)
    return A*math.exp(B/T+C*T+D*T*T)
#PVT block
def n2_viscosity_from_temp(T):
    mu_init = 1.7e-5
    T_init = 273
    S = 104.7
    return (mu_init*(T/T_init)**1.5)*(T_init+S)/(T+S)

#start of PVT block

rk_fluid = cubic('N2,ETOH', 'SRK')
T = 280 # Kelvin
p = 1e5 # Pascal
z = [0.5, 0.5] # Molar composition
m_ethanol = 0.03
m_n2 = 0.028
#x, y, vapfrac, liqfrac, phasekey = rk_fluid.two_phase_tpflash(T, p, z)
#print ("vap, liq, vapfrac, liqfrac, phasekey", x,y,vapfrac,liqfrac, phasekey)

rho_1, =  rk_fluid.specific_volume(T, p, z, 1)
rho_1 = m_ethanol/rho_1
rho_2, =  rk_fluid.specific_volume(T, p, z, 2)
rho_2 = m_n2/rho_2

rho_tot = z[0]*rho_1+z[1]*rho_2

mu_1 = eth_viscosity_from_temp(T)
mu_2 = n2_viscosity_from_temp(T)

v1 = 1 #[m/s]
D = 0.05 #[m]
L = 1000 #[m]
#end of PVT block
#start of regime part
xtt = calculate_xtt(rho_1, rho_2, mu_1, mu_2, v1, D)
regime = return_regime(xtt)
print ('Found ', regime, ' regime! ')

#end of regime part
#start of friction part
ff = return_friction_factor(xtt)

mu_tot = calculate_visc(mu_1, mu_2, ff)

Re = calculate_Re(v1,D, rho_tot, mu_tot)

print ('Reynolds number is ', Re)

P1 = p-return_pressure_loss(v1,D,L, return_lambda(Re), rho_tot)

print ('New pressure is', P1)
#pressure loss calculated, now time for calculating regime again
rho_1_1, =  rk_fluid.specific_volume(T, P1, z, 1)
rho_1_1= m_ethanol/rho_1_1
rho_2_1, =  rk_fluid.specific_volume(T, P1, z, 2)
rho_2_1 = m_n2/rho_2_1
rho_tot_1 = z[0]*rho_1+z[1]*rho_2

xtt = calculate_xtt(rho_1_1, rho_2_1, mu_1, mu_2, v1, D)
regime = return_regime(xtt)
print ('New regime is ', regime, ' ! ')
wss = 0.25*(p-P1)*D/L
print ('Average WSS is about', wss, 'Pa')