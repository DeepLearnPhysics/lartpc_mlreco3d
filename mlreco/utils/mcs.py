import scipy
import pint
import numpy as np
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


c=299792458*ureg.m/ureg.s
rho=1.396*ureg.g/ureg.cm**3
NA=6.0221408e+23/ureg.mol
re=2.81794*pow(10,-15)*ureg.meter
E0e=0.510998918*ureg.MeV
me=E0e/c**2
Z=18
A=39.948*ureg.gram/ureg.mol
I=188.0*ureg.eV
a=0.19559

cbar=5.2146
x0=.2
x1=3

# I=16*pow(Z,0.9)*ureg.eV

K=4*np.pi*NA*re**2*E0e

k=3


def beta(p, M):
    return p/np.sqrt(p*p+M*M*c*c)


def KEtoP(KE,M):
    # return np.sqrt((KE+M)**2-M*M)
    # print(KE,M)
    return np.sqrt(KE*KE+2*M*KE*c*c)/c


def PtoKE(p, M):
    return np.sqrt(p*p*c*c+M*M*c**4)-M*c*c


assert 500*ureg.MeV/c == KEtoP(PtoKE(500*ureg.MeV/c, 100*ureg.MeV/c**2), 100*ureg.MeV/c**2)


def gamma(beta):
    return 1/np.sqrt(1-beta**2)


def delta(gb): #density correction for Bethe_Bloch
    x = np.log10(gb)
    # if x >= x1: return 2*np.log(10)*x-cbar
    # if x0 <= x and x < x1: return 2*np.log(10)*x-cbar+a*(x1-x)**k
    # if x < x0: return 0
    return (2*np.log(10)*x-cbar+(a*(x1-x)**k)*(x<x1))*(x>x0)


def dEdx(E, M):
    assert E.dimensionality == (1*ureg.MeV).dimensionality
    assert M.dimensionality == (1*ureg.MeV/c**2).dimensionality

    if M == 0:
        raise Exception("Bad M", beta, M)
    bet = beta(KEtoP(E, M), M)
    gb = gamma(bet)*bet
    Wmax = 2*E0e*(gb**2)/(1+2*gamma(bet)*me/M+(me/M)**2)
    return -rho*K*Z/A/bet**2*(.5*np.log(2*E0e*(gb**2)*Wmax/I**2)-bet**2-delta(gb)/2)


def KE0_to_momentum_list(KE0,dx,M): #returns the momentum after every step of dx
    assert KE0.dimensionality== (1*ureg.MeV).dimensionality
    assert dx.dimensionality== (1*ureg.mm).dimensionality
    assert M.dimensionality== (1*ureg.MeV/c**2).dimensionality
    sub=10
    Elist=[KE0]
    dx0=dx/sub
    i=0
    while Elist[-1]!=0*ureg.MeV:
        i+=1
        if i%1000==0:print(i,Elist[-1],end="\r")
        E=Elist[-1]
        mydedx=dEdx(E,M)
        newE=E+dx0*mydedx
        if newE<0*ureg.MeV or mydedx>0*ureg.MeV/ureg.mm: break
        Elist+=[newE]
    Elist+=[0*ureg.MeV]
    Elist=Q_.from_list(Elist)
    plist=KEtoP(Elist,M)
    ret=plist[0::sub]
    if ret[-1]==0*ureg.MeV/c: return ret
    return np.append(plist[0::sub],0)

def splitangle(test):

    sign1=np.random.choice([-1,1],len(test))
    sign2=np.random.choice([-1,1],len(test))
    randt=np.random.uniform(0, 2*np.pi,len(test))

    t1=np.arctan(np.sin(randt)*np.tan(np.arccos(test)))*sign1
    t2=np.arctan(np.cos(randt)*np.tan(np.arccos(test)))*sign2

    return t1,t2


def fitfun(xx, aa, bb, cc):
    # print("start",xx,aa,"end help me pls")
    
    return np.power(1-np.exp(-aa*(xx/ureg.MeV*c).to_reduced_units().m), bb)*cc

def ptox(p,pxstore,mydxlist):
    assert p.dimensionality==(1*ureg.MeV/c).dimensionality
    # print(p,pxstoremu,mydxlist)
    # print("broken",p,pxstoremu,mydxlist,np.interp(p,pxstoremu[::-1],mydxlist[::-1]))
    return np.interp(p,pxstore[::-1],mydxlist[::-1])


def sigmathetasimp(p, M, L,al=np.inf,bl=0,cl=1):
    corr=fitfun(p,al,bl,cl)*ureg.MeV/c
    return np.sqrt(L/(140*ureg.mm))*np.divide(corr,p*beta(p, M))

def LLoss(P, thetalist1, thetalist2, M, L, AL, BL, CL, px, dx):

    assert px.dimensionality== (1*ureg.MeV/c).dimensionality
    assert dx.dimensionality== (1*ureg.mm).dimensionality
    assert M.dimensionality== (1*ureg.MeV/c**2).dimensionality
    assert L.dimensionality== (1*ureg.mm).dimensionality

    if len(thetalist1) <= 1: return np.inf*ureg.dimensionless
    ls = [i*L for i in range(len(thetalist2)+1)]
    plist = np.interp(ls, dx-ptox(P*ureg.MeV/c, px, dx),px,right=0*ureg.MeV/c)
    p = np.sqrt(plist[1:]*plist[:-1])
    st = sigmathetasimp(p, M, L, al=AL,bl=BL,cl=CL)
    ret = np.sum(.5*(thetalist1/st)**2+.5*(thetalist2/st)**2+2*np.log(st))
    return ret


def fitp(theta,M,L,AA,BB,CC,px,dx,pmin,pmax):#returns in Mev/c 

    assert px.dimensionality== (1*ureg.MeV/c).dimensionality
    assert dx.dimensionality== (1*ureg.mm).dimensionality
    assert M.dimensionality== (1*ureg.MeV/c**2).dimensionality
    assert L.dimensionality== (1*ureg.mm).dimensionality

    theta1,theta2=splitangle(theta)

    pmind = (pmin/(ureg.MeV/c)).to_base_units().m
    pmaxd=(pmax/(ureg.MeV/c)).to_base_units().m
    mymin=scipy.optimize.minimize_scalar(LLoss, np.array([1000]),args=(theta1,theta2,M,L,AA,BB,CC,px,dx),bounds=[pmind,pmaxd])
    return mymin.x*ureg.MeV/c



