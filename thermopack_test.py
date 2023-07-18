import sys
import math
import copy
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0,'../pycThermopack/')
from thermopack.cubic import cubic
from thermopack.saftvrmie import saftvrmie

@dataclass
class tube_point:
    name: str = "mixture"  # name: string, name of the material
    phase_name: List[str] = field(default_factory=lambda: ["gas"]) #phase_name: list of strings, name of phases
    number_of_fluids: int = 1  # number_of_fluids: integer, number of phases
    T: float = 300.0  # T: double, local temperature, Kelvin
    p: float = 101325.0  # p: double, local pressure, Pascal
    v: float = 0.5 #v: double, mixture velocity, m/s
    D: float = 0.1 #D: double, tube local diameter, m
    L: float = 10 #L: double, tube length, m. Zero if it's endpoint
    section_type = 1 #section type: integer, type of the section for losses calculation, not used yet - enum!
    molar_composition: List[float] = field(default_factory=lambda: [1.0])  # molar_composition: list of doubles, molar composition [probably constant]
    molar_masses: List[float] = field(default_factory=lambda: [1.0])  # molar_masses: list of doubles, molar masses [constant overall]
    vapor_components: List[float] = field(default_factory=lambda: [0.5])  # vapor_components: list of doubles, vapor distribution over components (sum is 1)
    liquid_components: List[float] = field(default_factory=lambda: [0.5])  # liquid_components: list of doubles, liquid distribution over components (sum is 1)
    components_density: List[float] = field(default_factory=lambda: [1.0]) # components_density: list of doubles, density distribution over components
    overall_density:float = 1.0 #overall_density: double, overall density, kg/m^3
    overall_vapor_fraction: float = 0.5  # overall_vapor_fraction: double, vapor distribution over mixture
    overall_liquid_fraction: float = 0.5  # overall_liquid_fraction: double, liquid distribution over mixture
    liquid_viscosities: List[float] = field(default_factory=lambda: [1.0e-3])  # liquid_viscosities: list of doubles, viscosity of liquid parts over components
    vapor_viscosities: List[float] = field(default_factory=lambda: [1.0e-3])  # vapor_viscosities:  list of doubles, viscosity of vapor parts over components
    liquid_overall_viscosity: float = 1.0e-3  # liquid_overall_viscosity: double, viscosity of liquid part
    vapor_overall_viscosity: float = 1.0e-3  # vapor_overall_viscosity: double, viscosity of vapor part
    overall_viscosity: float = 1.0e-3  # overall_viscosity: double, viscosity of mixture
    regime: str = "bubble"  # regime: string, name of selected flow regime
    regime_key: float = 1.0  # regime_key: double, currently XTT, later - other number to characterize regime
    regime_friction_factor: float = 1.0  # regime_friction_factor: double, currently from XTT
    reynolds_number: float = 10000.0  # reynolds_number: double, Reynolds number for ...
       
    

def calculate_xtt (rho1, rho2, mu1, mu2, v1, D):
    return ((1.096/rho1)**0.5)*((rho1/rho2)**0.25)*((mu2/mu1)**0.1)*((v1/D)**0.5)

def calculate_xtt (point: tube_point):
    rho1 = point.components_density[0]
    rho2 = point.components_density[1]
    mu1 = point.liquid_viscosities[0]
    mu2 = point.liquid_viscosities[1]
    v1 = point.v
    D = point.D
    return ((1.096/rho1)**0.5)*((rho1/rho2)**0.25)*((mu2/mu1)**0.1)*((v1/D)**0.5)


def calculate_visc (mu1, mu2, ff):
    return ff*mu1+(1-ff)*mu2

def calculate_Re (V,D,rho, mu):
    return V*D*rho/mu

def calculate_Re (point: tube_point):
    V = point.v
    D = point.D
    rho = point.overall_density
    mu = point.overall_viscosity
    return V*D*rho/mu


def return_lambda(Re):
    if (Re<2300): return 64/Re
    else: return 0.316/(Re**0.25)

def return_pressure_loss(V,D,L, lam, rho):
    xi = lam*L/D
    return (xi*V**2)*0.5*rho

def get_overall_density(point:tube_point):
    dens = 0.0
    for i in range(point.number_of_fluids):
        dens += point.molar_composition[i]*point.components_density[i]
    return dens

def return_regime (xtt):
        if(xtt<10): return 'bubble'
        if(xtt>=10 and xtt<100): return 'plug'
        if(xtt>=100 & xtt<1000): return 'slug'
        if(xtt >= 1000 & xtt<10000): return 'annular'
        if (xtt>=10000): return 'mist'
        return 'undefined'
    
#liquid to solid viscosity calculation:

def return_friction_factor (xtt): 
        if(xtt<10): return 1
        if(xtt>=10 and xtt<100): return 0.9
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

def start_point (point: tube_point): #initialization of start point, done by hand
    rk_fluid = cubic('N2,ETOH', 'SRK') #obsolete
    point.T = 320.0 # Kelvin
    T = point.T
    point.p = 3.5*101325 # Pascal
    point.molar_composition = [0.5, 0.5] # Molar composition
    point.molar_masses = [0.03, 0.028] 

    x, y, vapfrac, liqfrac, phasekey = rk_fluid.two_phase_tpflash(point.T, point.p, point.molar_composition)
    #print ("vap, liq, vapfrac, liqfrac, phasekey", x,y,vapfrac,liqfrac, phasekey)
    point.vapor_components = x
    point.liquid_components = y
    point.vapfrac = vapfrac
    point.liqfrac = liqfrac
    
    rho_1, =  rk_fluid.specific_volume(point.T, point.p, point.molar_composition, 1)
    rho_1 = point.molar_masses[0]/rho_1
    rho_2, =  rk_fluid.specific_volume(point.T, point.p, point.molar_composition, 2)
    rho_2 = point.molar_masses[1]/rho_2
    point.components_density = [rho_1, rho_2]
    point.overall_density = get_overall_density(point)
        
    mu_1 = eth_viscosity_from_temp(T)
    mu_2 = n2_viscosity_from_temp(T)
    
    point.liquid_viscosities = [mu_1, mu_2]
    point.vapor_viscosities = [mu_1, mu_2]
    
    point.v = 5.0 #[m/s]
    point.D = 0.08 #[m]
    point.L = 100
    return point
    #v1 = 5 #[m/s]
    #D = 0.08 #[m]
    #L = 1000 #[m]

def define_tube_params(point: tube_point, D, L, rho_old):
    q = point.v*point.D*point.D*rho_old
    V_new = q/(D*D*point.overall_density) #mass balance, pi/4 is skipped 
    #due to presence in both parts of equation
    point.D = D
    point.L = L
    point.V = V_new
    return point

def pvt_block(point: tube_point, new_P, new_T):
    rk_fluid = cubic('N2,ETOH', 'SRK') #obsolete
    point.T = new_T
    point.p = new_P
    x, y, vapfrac, liqfrac, phasekey = rk_fluid.two_phase_tpflash(new_T, new_P, point.molar_composition)
    point.vapor_components = x
    point.liquid_components = y
    point.vapfrac = vapfrac
    point.liqfrac = liqfrac
    rho_1, =  rk_fluid.specific_volume(new_T, new_P, point.molar_composition, 1)
    rho_1 = point.molar_masses[0]/rho_1
    rho_2, =  rk_fluid.specific_volume(new_T, new_P, point.molar_composition, 2)
    rho_2 = point.molar_masses[1]/rho_2
    point.components_density = [rho_1, rho_2]
    point.overall_density = get_overall_density(point)
    mu_1 = eth_viscosity_from_temp(new_T)
    mu_2 = n2_viscosity_from_temp(new_T)
    point.liquid_viscosities = [mu_1, mu_2]
    point.vapor_viscosities = [mu_1, mu_2]
    return point
    
p = tube_point()
start_point(p)
tube_D = [0.08, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.1]
tube_L = [1000, 200, 450, 1000, 200, 300, 200, 300, 1000, 0]
tube_points = [p]
for i in range(1, len(tube_D)):
    xtt = calculate_xtt(tube_points[i-1])
    #print ('i', i, ' xtt ', xtt)
    regime = return_regime(xtt)
    print ('Found ', regime, ' regime at', i,'! xtt= ', xtt)
    ff = return_friction_factor(xtt)
    tube_points[i-1].overall_viscosity = calculate_visc(tube_points[i-1].liquid_viscosities[0], tube_points[i-1].liquid_viscosities[1], ff)
    tube_points[i-1].Re = calculate_Re(tube_points[i-1])
    print ('Reynolds number for ', i, ' is ', tube_points[i-1].Re, 'lambda is ', return_lambda(tube_points[i-1].Re))
    diff = return_pressure_loss(tube_points[i-1].v,tube_points[i-1].D,tube_points[i-1].L, return_lambda(tube_points[i-1].Re), tube_points[i-1].overall_density)
    P1 = tube_points[i-1].p-diff
    T1 = tube_points[i-1].T-i*0.3
    print(P1, T1)
    p2 = copy.deepcopy(tube_points[i-1])
    rho = p2.overall_density
    p2 = pvt_block(p2,P1,T1)
    p2 = define_tube_params(p2, tube_D[i], tube_L[i], rho)
    tube_points.append(p2)
#end of PVT block
#start of regime part
#xtt = calculate_xtt(p)
#regime = return_regime(xtt)
#print ('Found ', regime, ' regime!; xtt= ', xtt)
#end of regime part
#start of friction part
#ff = return_friction_factor(xtt)

#p.overall_viscosity = calculate_visc(p.liquid_viscosities[0], p.liquid_viscosities[1], ff)

#p.Re = calculate_Re(p)

#print ('Reynolds number is ', p.Re)
#diff = return_pressure_loss(p.v,p.D,L, return_lambda(p.Re), p.overall_density)
#print (p.p, diff)
#P1 = p.p-diff
#T1 = p.T-20.0
#print ('New pressure is', P1)

#p2 = copy.deepcopy(p)
#pvt_block(p2, P1, T1)
#xtt2 = calculate_xtt(p2)
#regime2 = return_regime(xtt2)
#print ('New regime is ', regime2, ' !; xtt= ', xtt2)
#pressure loss calculated, now time for calculating regime again
#print ('dp', p.p, P1, 'D', p.D, 'L', L)
#wss = 0.25*(diff)*p.D/L
#print ('Average WSS is about', wss, 'Pa')

#tube = [p,p2]

#rho_1_1, =  rk_fluid.specific_volume(T1, P1, z, 1)
#rho_1_1= m_ethanol/rho_1_1
#rho_2_1, =  rk_fluid.specific_volume(T1, P1, z, 2)
#rho_2_1 = m_n2/rho_2_1
#rho_tot_1 = z[0]*rho_1+z[1]*rho_2


#xtt = calculate_xtt(rho_1_1, rho_2_1, mu_1, mu_2, v1, D)
#regime = return_regime(xtt)
#print ('New regime is ', regime, ' !; xtt= ', xtt)
#wss = 0.25*(p-P1)*D/L
#print ('Average WSS is about', wss, 'Pa')