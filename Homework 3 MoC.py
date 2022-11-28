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
Edge = SurfY[0], SurfY[-1], SurfX[0], SurfX[-1]     #The set of edges to the space - the first and last X and Y surface


#Rotate Surfaces
def RotateSurfaces(Surfaces, Angle):
    #Surfaces: the set of surfaces to rotate
    #Angle: the angle to rotate the surfaces
    #Augments the quadratic parameters of the surface
    #Requires special treatment for vertical lines (AY+BX+C=0; A = 0), going into and out of veritcal lines
    for iSurface in Surfaces:
        A1, B1, C1 = iSurface.WriteABC()
        if B1 == 0:
            ThetaN0 = 0
        else:
            ThetaN0 = np.arctan(1/B1)
        
        
        if A1 == 0:     #Case 1: Starting line is vertical, treat as creating new line
            A2 = 1
            B2 = 1/np.tan(Angle)
            C2 = C1 / np.sin(Angle)
        
        
        elif (ThetaN0 + Angle)%np.pi == 0.0     #Case 2: Final Line is vertical, 
            A2 = 0
            B2 = 1
            C2 = C1 * np.sin(ThetaN0)        
        
        
        else:   #Final Case, neither beginning nor ending line is vertical
            A2 = A1
            B2 = (B1 - np.tan(Angle))/(1+np.tan(Angle))
            C2 = C1 / (np.cos(Angle) + B1 * np.sin(Angle))
        iSurface.A = A2
        iSurface.B = B2
        iSurface.C = C2


#Find Corners

def FindCorners(Sy1, Sy2, Sx1, Sx2):
    #For four perpendicular lines that create a rectangular shape, returns the xy coordinates of that shape
    Yset = []
    Xset = []
    Y, X = 0.0, 0.0
    for iSy in [Sy1, Sy2]:
        for iSx in [Sx1, Sx2]:
            Y,X = FindIntersection(iSy, iSx)
            Yset.append(Y)
            Xset.append(X)
    return Yset, Xset        

def FindIntersection(Sy, Sx):
    #Two surfaces that should be orthogonal to each other
    if Sy.A == 0 and Sx.B == 0: #Sy is vertical, Sx is horizontal
        X = -1*Sy.C
        Y = -1*Sx.C
    
    
    elif Sy.B == 0 and Sx.A == 0: #Sx is vertical, Sy is horizontal (ie problem rotated 90 degrees)
        X = -1*Sx.C
        Y = -1*Sy.C
    
    
    elif Sy.B * Sx.B == -1    :   #problem is not orthogonal to cartesian plane, but surfaces are orthogonal to each other
        Mat = [[Sy.A, Sy.B],[Sx.A, Sx.B]]
        Cvec = [-1*Sy.C, -1*Sx.C]
        Y,X = np.linalg.inv(Mat) * Cvec
    
    else:
        print("Error, Surfaces are not orthogonal")
    return Y,X

        
        

#Plot Rotating Lines


#Find Sweep Start Position (surface)
#Find Sweep start position (cell)

#Find Next Sweep Position (surfaces touching cell)

#Generate Angular flux contribution for given cell

#Store thicknesses and cells

#Repeat sweep for opposite direction using stored thicknesses and cells
    
    
    
    
