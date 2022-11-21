import numpy as np
import matplotlib.pyplot as plt

class Cell:     #Cell([T,S0,S1,SA,Sf],[X0,X1, Y0,Y1, Z0,Z1],Q)
    "Information for an individual 1D Cell"
    def __init__(self, Sig = [1.0, 0.0, 0.0, 1.0, 0.0], Geo = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], Qsrc = 0.0):
        self.SigT   = Sig[0]    #Total Macroscopic Cross Section
        self.SigS0  = Sig[1]    #Isotropic Scattering
        self.SigS1  = Sig[2]    #Linear Anisotropic scattering 
        self.SigA   = Sig[3]    #Absorption
        self.nuSigF = Sig[4]    #Fission Multiplication
        self.X0     = Geo[0]    #Location of left boundary
        self.X1     = Geo[1]    #Location of right boundary
        self.Y0     = Geo[2]    #Location of left boundary
        self.Y1     = Geo[3]    #Location of right boundary
        self.Z0     = Geo[4]    #Location of left boundary
        self.Z1     = Geo[5]    #Location of right boundary
        self.Qsrc   = Qsrc
        self.Df     = 1.0 / (self.SigT - self.SigS0) / 3.0     #Diffusion Coefficient
        self.Linv   = self.SigT / self.Df       #1/L^2
        self.Phi    = 0.0      #Isotropic Flux     
        
    def SIG(self):
        return [self.SigT, self.SigS0, self.SigS1, self.SigA, self.nuSigF]
    
    def GEO(self):
        return [self.X0, self.X1, self.Y0, self.Y1, self.Z0, self.Z1]
        
    def DelR (self):    #Returns dX, dY, dZ
        return self.X1 - self.X0, self.Y1 - self.Y0,  self.Z1 - self.Z0

class Material:
    def __init__ (self, Sig = [1.0, 0.0, 0.0, 1.0, 0.0], Qsrc = 0.0):
        self.SigT   = Sig[0]    #Total Macroscopic Cross Section
        self.SigS0  = Sig[1]    #Isotropic Scattering
        self.SigS1  = Sig[2]    #Linear Anisotropic scattering 
        self.SigA   = Sig[3]    #Absorption
        self.nuSigF = Sig[4]    #Fission Multiplication
        self.Qsrc   = Qsrc
    def SIG(self):
        return [self.SigT, self.SigS0, self.SigS1, self.SigA, self.nuSigF]        
        
class SubLine:
    def __init__(self, Thickness, Material, CellCount):
        self.Width = Thickness
        self.Mat = Material
        self.XCell = CellCount 
    def dX (self):
        return self.Width / self.XCell
        
class SubLayer:
    def __init__ (self, Thickness, Line, CellCount):
        self.Depth = Thickness
        self.Line = Line
        self.YCell = CellCount
    def dY (self):
        return self.Depth / self.YCell
        
class SubVolume:
    def __init__ (self, Thickness, Layer, CellCount):
        self.Height = Thickness
        self.Layer = Layer
        self.ZCell = CellCount
    def dZ (self):
        return self.Height / self.ZCell

def Create2DSpace(Layer):
    x = 0.0
    y = 0.0
    Space = []
    for sublayer in Layer:
        DY = sublayer.dY()
        for yCell in range(sublayer.YCell):
            for subline in sublayer.Line:
                DX = subline.dX()
                #print(DX)
                Sig = subline.Mat.SIG()
                Q = subline.Mat.Qsrc
                for xCell in range(subline.XCell):
                    Geo = [x,x+DX, y, y+DY, 0.0, 1.0]
                    Space.append(Cell(Sig,Geo,Q))
                    x += DX
            x = 0.0
            y += DY
            
            
    return Space

def Create3DSpace(Volume):
    x = 0.0
    y = 0.0
    z = 0.0
    Space = []
    for subvolume in Volume:
        DZ = subvolume.dZ()
        for zCell in range(subvolume.ZCell):
            for sublayer in subvolume.Layer:
                DY = sublayer.dY()
                for yCell in range(sublayer.YCell):
                    for subline in sublayer.Line:
                        DX = subline.dX()
                        #print(DX)
                        Sig = subline.Mat.SIG()
                        Q = subline.Mat.Qsrc
                        for xCell in range(subline.XCell):
                            Geo = [x,x+DX, y, y+DY, z, z+DZ]
                            Space.append(Cell(Sig,Geo,Q))
                            x += DX

                            
                    x = 0.0
                    y += DY
            y = 0.0
            z += DZ
            
    return Space
   
def Create3DMatrix(Space, N, BCs):
#Creates the Matrix (M) for solving the diffusion equation and returns its inversion
#M*phi = Q
#phi = M^-1 * Q
#Space is a set of cells
#N is a vector consisting of the number of cells in the X, Y and Z directions
#BCs are a set of boundary conditions: BC[E, W, N, S, U, D]
#BC options are vacuum (-1), mirror (1) 
    Nx, Ny, Nz = 1,1,1
    Face    = ['East', 'West', 'North', 'South', 'Up', 'Down']
    
    if len(N) == 3:
        Nx, Ny, Nz = N
    else:
        print("Invalid N Cell Data")
        return 0
    Shift   = [1,-1, Nx, -1*Nx, Nx*Ny, -1*Nx*Ny]    
    M = np.zeros([Nx*Ny*Nz,Nx*Ny*Nz])
    Q = np.zeros(Nx*Ny*Nz)
        
    for iZ in range(Nz): 
        for iY in range(Ny):
            for iX in range(Nx):
                l = iX + iY * Nx + iZ * Nx * Ny
                DX, DY, DZ = Space[l].DelR()
                DR = [DX, DX, DY, DY, DZ, DZ]
                DC  = Space[l].Df

                Alpha = 0
                Beta = 0
                Bound = [iX == Nx -1, iX == 0, iY == Ny -1, iY == 0, iZ == Nz -1, iZ == 0]
                
                #For each face, calculate Alpha and Beta
                #print(l, iX, iY, iZ)

                for f in range(6):
                    if   Bound[f] and BCs[f] == -1:       #Vacuum Correction East
                        #Alpha = Beta = 0.0
                        DR[f] += 0.71 * DC                     
                    
                    elif Bound[f] and BCs[f] == 1:       #Mirror Correction
                        #Alpha = 0
                        Beta = 1.0
                    
                    elif not Bound[f]:
                        DI = Space[l+Shift[f]].Df
                        XI = Space[l+Shift[f]].X1 - Space[l+Shift[f]].X0
                        XC = DR[f]
                        #DC = DiffCen
                        
                        
                        Alpha = DI * XC / (DC * XI + DI * XC)
                        Beta  = DC * XI / (DC * XI + DI * XC)
                        
                        M[l,l+Shift[f]] = Alpha / DR[f]**2

                        if l+Shift[f] >= Nx*Ny*Nz or l+Shift[f] < 0:
                            print("Error found at point:")
                            print("l, Face, iX, iY, iZ")
                            print(l, Face[f], iX, iY, iZ)
                            print("Nx, Ny, Nz")
                            print(Nx, Ny, Nz)
                    
                     
                    M[l,l] += (Beta-1) / DR[f]**2
                #Calculate Mc
                M[l,l] -= Space[l].Linv
                Q[l]    = -1 * Space[l].Qsrc / DC
 
    #return np.linalg.inv(M), Q
    return M, Q

def GenerateCube(MatChoice, SetChoice, InnerN, BarrierN):

    Reflector   = Material([2.00, 1.80 , 0.0, 0.20 , 0.0], 0.0)
    Scatterer   = Material([2.00, 1.99 , 0.0, 0.01 , 0.0], 0.0)
    Absorber    = Material([10.0, 2.00 , 0.0, 8.00 , 0.0], 0.0)
    Air         = Material([0.01, 0.006, 0.0, 0.004, 0.0], 0.0)
    Isotropic   = Material([0.10, 0.0  , 0.0, 0.0  , 0.0], 1.0)  #Qsrc = 1.0
      
    MatDict = {
        "Ref" : Reflector,
        "Sca" : Scatterer,
        "Abs" : Absorber ,
        "Air" : Air      
        }
    BarrierMat = MatDict[MatChoice]
    #Geometry
    DSet        = [0.01, 0.1, 1.0 , 10 ]
    WSet        = [0.1 , 1.0  , 10.0, 100]
    InnerD = DSet[SetChoice]
    OuterW = WSet[SetChoice]
    BarrierW = (OuterW - InnerD) / 2.0
    #Cell counts
    OuterN = InnerN + 2 * BarrierN
    N = [OuterN,OuterN,OuterN]
    #Boundary Conditions
    #  0: Vacuum
    # +1: Mirror
    # [East, West, North, South, Up, Down]
     #WWWW   
     #DQQD   
     #DQQD   
     #WWWW   
    #Barrier SubLine    
    SLineW1 = SubLine(BarrierW, BarrierMat, BarrierN)
    SLineW2 = SubLine(InnerD, BarrierMat, InnerN)
    #Edge Subline
    SLineD = SubLine(BarrierW , BarrierMat, BarrierN)
    #Meat SubLine
    SLineQ = SubLine(InnerD, Isotropic, InnerN)


    #Lines
    Line1 = [SLineW1, SLineW2, SLineW1]
    Line2 = [SLineD , SLineQ , SLineD ]


    #SubLayers
    SLayer1 = SubLayer(BarrierW, Line1, BarrierN)
    SLayer2 = SubLayer(InnerD, Line2, InnerN)

    SLayerB1 = SubLayer(BarrierW, Line1, BarrierN)
    SLayerB2 = SubLayer(InnerD, Line1, InnerN)
    #Layer
    Layer1 = [SLayer1,SLayer2,SLayer1]

    LayerB = [SLayerB1, SLayerB2, SLayerB1]
    #SubVolumes
    SubVol1 = SubVolume(1.0, Layer1, 1)


    SubVolB = SubVolume(BarrierW, LayerB, BarrierN)
    SubVolD = SubVolume(InnerD, Layer1, InnerN)

    #Volumes

    #Single Layer Volume:
    Vol1 = [SubVol1]

    #The Cube:
    Cube = [SubVolB, SubVolD, SubVolB]

    return Cube

def Solve3DFlux(Space, N, BCs):
    M_Cube, Q_Cube = Create3DMatrix(Space, N, BCs)
    M_Inv = np.linalg.inv(M_Cube)
    Flux = np.dot(M_Inv, Q_Cube)
    l = 0
    for cell in Space:
        cell.Phi = Flux[l]
        l+=1
    
    return Flux

def OuterAvg(Space):
#Assumes the Space has already had it's flux calculated
    OuterPhiAvg = 0
    N_out = 0
    for cells in Space:
        if cells.Qsrc == 0:     #Outer Material has no source
            OuterPhiAvg += cells.Phi
            N_out += 1
    return OuterPhiAvg / N_out

def GenerateResults(Space, Start, Stop, NR):
#Space has already had its flux calculated
#Start and Stop are a 3D cartesian points in space within Space
#NR is the number of points to traverse. Recomend very small compared to Space DV
    Results = []
    Distance = []
    X0, Y0, Z0 = Start
    X1, Y1, Z1 = Stop
    dx = (X1-X0) / NR
    dy = (Y1-Y0) / NR
    dz = (Z1-Z0) / NR
    Range  = ((X1-X0)**2 + (Y1-Y0)**2 + (Z1-Z0)**2)**0.5
    dRange = Range / NR
    R1 = 0
    #print("Range and dR:")
    #print(Range, dRange)
    X,Y,Z = Start
    for iR in range(NR+1):
        next = False
        l = 0
        while not next:
            cellX0, cellX1, cellY0, cellY1, cellZ0, cellZ1 = Space[l].GEO()
            if X >= cellX0 and X <= cellX1 and Y >= cellY0 and Y <= cellY1 and Z >= cellZ0 and Z <= cellZ1:     #Point is inside the cell
                Results.append(Space[l].Phi)
                Distance.append(R1)
                next = True
            
            l += 1
            if l >= len(Space) and next == False:
                print("Error: Out of Bounds at:")
                print("l, X, Y, Z")
                print(l,X, Y, Z)
                break
        X  += dx
        Y  += dy
        Z  += dz
        R1 += dRange 
        
    return Results , Distance
    



#Analysis
#Each material gets a data set
#For each data set, sensitivity study at n = 2, 4, 8 (total N 216, 12^3, 24^)
#For each sensitivity study, display the long diagonal and the bisection
#For each plot, display the average outside flux

#1 - Select Material and dimensions
SetChoice = 0
SetW        = 10 ** (SetChoice - 1)
#SetW = 2
MatChoice = 'Sca'
MatName = "Scatterer"
NPoints = 100

LongStart  = [0,0,0]
LongStop   = [SetW,SetW,SetW]
ShortStart = [0, SetW/2, SetW/2]
ShortStop  = [SetW, SetW/2, SetW/2]
BCList3D = [-1, -1, -1, -1, -1, -1]


PlotPhi = []
PlotX   = []
phiAvg  = []

NDisplay = []

#2 - Loop through N values
for iN in range(3):
    InnerN  = 2**(iN+1)
    BarrierN  = 2*(iN+1)
    


    OuterN = InnerN + BarrierN*2
    N = [OuterN,OuterN,OuterN]
    NDisplay.append(N[0]*N[1]*N[2])
    
    
#3 - For each N value, store Long and Short path, along with the average outside flux
    Cube = GenerateCube(MatChoice, SetChoice, InnerN, BarrierN)
    CubeSpace = Create3DSpace(Cube)
    Phi = Solve3DFlux(CubeSpace, N, BCList3D)
    phiAvg.append(OuterAvg(CubeSpace))
    #Long Results
    PhiLong, XLong = GenerateResults(CubeSpace, LongStart, LongStop, NPoints)
    PlotPhi.append(PhiLong)
    PlotX.append(XLong)
    #Short Results
    PhiShort, XShort = GenerateResults(CubeSpace, ShortStart, ShortStop, NPoints)
    PlotPhi.append(PhiShort)
    PlotX.append(XShort)
    
#4 - Create 3x2 plot, each of which has the AVG, Long diagonal on the left and short path on the right

for iPlt in range(3):
    iPlot = iPlt*2
    x1 = PlotX[iPlot]
    y1 = np.log(PlotPhi[iPlot])
    y2 = np.log([phiAvg[iPlt], phiAvg[iPlt]])
    x2 = [x1[0],x1[-1]]
    
    plt.subplot(3,2,iPlot+1)
    plt.plot(x1,y1, label="Phi")
    plt.plot(x2,y2, label="Phi_Avg")
    plt.xlabel("X [cm]")
    plt.ylabel("Log Flux [n/cm^2/sec]") 
    plt.title("Diagonal Flux with N = "+str(NDisplay[iPlt]))  
    plt.legend()
    
    iPlot = iPlt*2+1
    x1 = PlotX[iPlot]
    y1 = np.log(PlotPhi[iPlot])
    y2 = np.log([phiAvg[iPlt], phiAvg[iPlt]])
    x2 = [x1[0],x1[-1]]
    
    plt.subplot(3,2,iPlot+1)
    plt.plot(x1,y1, label="Phi")
    plt.plot(x2,y2, label="Phi_Avg")
    plt.xlabel("X [cm]")
    plt.ylabel("Log Flux [n/cm^2/sec]") 
    plt.title("Horizontal Flux with N = "+str(NDisplay[iPlt]))
    plt.legend()    
plt.suptitle(MatName+" Diffusion Plots", fontsize = 20)
plt.subplots_adjust(top=0.935, bottom=0.065, left=0.045, right=0.99, hspace=0.345, wspace=0.13)

plt.show()

