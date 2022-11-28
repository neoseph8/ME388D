import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Surface:
    def __init__ (self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        #And by surface we mean line, where A or B = 1
        #AY + BX + C = 0
        self.Cells = []
        self.Intercept = -1*C
    def WriteABC(self):
        return self.A, self.B, self.C
       
class Cell:
    def __init__(self, Material = [0,0,0,0,0] , Surfaces = [], QBase = 0):
        #Material = XSECs [ SigT, S0, S1, SA, nSF]
        self.SigT       = Material[0]
        self.SigS0      = Material[1]
        self.SigS1      = Material[2]
        self.SigA       = Material[3]
        self.SigF       = Material[4]
        self.Surfaces   = Surfaces      #X0, X1, Y0, Y1
        self.QBase      = QBase  

def GenerateRadials(Tq, Tb, Nq, Nb):
    dTq = Tq / Nq   #Thickness per source cell
    dTb = Tb / Nb   #Thickness per barrier cell
    r = []
    r.append(0) #First radial point
    for iN in range(Nq):
        r.append(dTq * (iN+1) )
    RB1 = r[-1]
    for iN in range(Nb):
        r.append(dTb * (iN+1) + RB1)
    return r

def GenerateSurfaces(radials):
    #Nr = 1 + 2 * len(radials)
    Sx = []
    Sy = []
    Sx.append(Surface(0,1,0))   #Zeroth X Line @ Y = 0
    Sy.append(Surface(1,0,0))   #Zeroth Y Line @ X = 0
   
    for iR in radials[1:len(radials)]:
        Sx.insert( 0,Surface(0,1,iR   ))    #Negative line prepended to front
        Sx.append(   Surface(0,1,-1*iR))    #Positive Line appended to back
        Sy.insert( 0,Surface(1,0,iR   ))    #Negative line prepended to front
        Sy.append(   Surface(1,0,-1*iR))    #Positive Line appended to back
   
    return Sx, Sy

def CreateSurfSpace(Surfaces, Nq, Nb, Materials):
    MatQ, MatB, Q = Materials
    Sx, Sy = Surfaces
    NCell = 2*(Nq+Nb)
    if len(Sx) != len(Sy) or len(Sx) != NCell+1:
        print("Surface Dimenension Errors!")
        return 0
       
    Y = -1 * (Tq + Tb)
    X = -1 * (Tq + Tb)
   
    Space = []
   
    for iY in range(NCell):
       
        for iX in range(NCell):
            MySurface = Sx[iX], Sx[iX+1], Sy[iY], Sy[iY+1]
           
            if iX >= Nb and iX < Nb + 2 * Nq and iY >= Nb and iY < Nb + 2 * Nq: #Inside the Source
                MyMat   = MatQ
                Qsrc    = Q
            else:
                MyMat   = MatB
                Qsrc    = 0.0
            MyCell = Cell(MyMat, MySurface, Qsrc)
            Space.append(MyCell)
           
            Sx[iX].Cells.append(MyCell)
            Sx[iX+1].Cells.append(MyCell)
            Sy[iY].Cells.append(MyCell)
            Sy[iY+1].Cells.append(MyCell)
    return Space        



Tq  = 1     #Source Thickness
Tb  = 9     #Barrier Thickness
Nq  = 1     #Number of source cells
Nb  = 1     #Number of Barrier cells

Rads = (GenerateRadials(Tq, Tb, Nq, Nb))
SurfX, SurfY = GenerateSurfaces(Rads)
print("A, B, C\nX Lines:")
for surf in SurfX:
    print(surf.A, surf.B, surf.C)
print("Y Lines:")
for surf in SurfY:
    print(surf.WriteABC())
   
MaterialQ   = [1,0,0,1,0]
MaterialB   = [1,0.5,0.1,0.4,0]
Qsrc        = 1

MaterialSet = MaterialQ, MaterialB, Qsrc

SurfSpace = CreateSurfSpace([SurfX, SurfY], Nq, Nb, MaterialSet)
l = 0
for cells in SurfSpace:
    print(l, cells.QBase, end = " ")
    for surfs in cells.Surfaces:
        print(surfs.Intercept, end = " ")
    print("")
    l+=1
    
#Define Outer Edge Surfaces

#Rotate Surfaces

#Find Sweep Start Position (surface)
#Find Sweep start position (cell)

#Find Next Sweep Position (surfaces touching cell)

#Generate Angular flux contribution for given cell

#Store thicknesses and cells

#Repeat sweep for opposite direction using stored thicknesses and cells
    
    
    
    
