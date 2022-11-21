import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Cell:     #Cell([T,S0,S1,SA,Sf],[X0,X1, Y0,Y1, Z0,Z1],Q)
    "Information for an individual 1D Cell"
    def __init__(self, Sig = [1.0, 0.0, 0.0, 1.0, 0.0], Geo = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], Qsrc = 0.0):
        self.SigT    = Sig[0]    #Total Macroscopic Cross Section
        self.SigS0   = Sig[1]    #Isotropic Scattering
        self.SigS1   = Sig[2]    #Linear Anisotropic scattering 
        self.SigA    = Sig[3]    #Absorption
        self.nuSigF  = Sig[4]    #Fission Multiplication
        self.X0      = Geo[0]    #Location of left boundary
        self.X1      = Geo[1]    #Location of right boundary
        self.Y0      = Geo[2]    #Location of left boundary
        self.Y1      = Geo[3]    #Location of right boundary
        self.Z0      = Geo[4]    #Location of left boundary
        self.Z1      = Geo[5]    #Location of right boundary
        self.Qsrc    = Qsrc
        self.QScat   = []        #Scattering Source (one for each angle)
        self.QAll    = []        #Scatter + Intrinsic Source
        self.Df      = 1.0 / (self.SigT - self.SigS0) / 3.0     #Diffusion Coefficient
        self.Linv    = self.SigT / self.Df       #1/L^2
        self.Phi00   = 0.0       #Isotropic Flux            (l,m = 0,0)
        self.Phi10   = 0.0       #Linear Anisotropic Flux 1 (l,m = 1,0)
        self.Phi11   = 0.0       #Linear Anisotropic Flux 2 (l,m = 1,1)
        self.PhiOld       = 0.0       #Place Holder for convergence testing
        self.PsiEdge = [0.0, 0.0, 0.0, 0.0]     #East, West, North, South angular fluxes
        self.PsiC    = []        #Central Angular Flux (one for each angle)
        
    def SIG(self):
        return [self.SigT, self.SigS0, self.SigS1, self.SigA, self.nuSigF]
    
    def GEO(self):
        return [self.X0, self.X1, self.Y0, self.Y1, self.Z0, self.Z1]
        
    def DelR (self):    #Returns dX, dY, dZ
        return self.X1 - self.X0, self.Y1 - self.Y0,  self.Z1 - self.Z0
        
    def ResetAngle (self, N_Omega):
        #Sets the size of the QScat and PsiC vectors based on the number of Discrete Ordinates, and adds intrinsic source / M to QAll
        M = N_Omega * 4
        for iM in range (M):
            self.QScat.append(0.0)
            self.PsiC.append(0.0)
            self.QAll.append(self.Qsrc / M)

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

def GenerateSquare(MatChoice, SetChoice, InnerN, BarrierN):

    Reflector   = Material([2.00, 1.80 , 0.0, 0.20 , 0.0], 0.0)
    Scatterer   = Material([2.00, 1.99 , 0.0, 0.01 , 0.0], 0.0)
    Absorber    = Material([10.0, 2.00 , 0.0, 8.00 , 0.0], 0.0)
    Air         = Material([0.01, 0.006, 0.0, 0.004, 0.0], 0.0)
    Isotropic   = Material([0.10, 0.0  , 0.0, 0.1  , 0.0], 1.0)  #Qsrc = 1.0
      
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

    #Layer
    Layer1 = [SLayer1,SLayer2,SLayer1]
    
    return Layer1

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

def Solve3DDiffusion(Space, N, BCs):
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
    #Returns volume averaged scalar flux
    OuterPhiAvg = 0.0
    V_Tot = 0.0
    for cells in Space:
        DX, DY, DZ = cells.DelR()
        CellVol = DX*DY*DZ
        V_Tot+=CellVol
        if cells.Qsrc == 0:     #Outer Material has no source
            OuterPhiAvg += cells.Phi00 * CellVol

    return OuterPhiAvg / V_Tot

def Generate3DResults(Space, Start, Stop, NR):
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
    
def GenerateQuadrature(N_Omega):
    #This function returns a set of 2D quadrature corresponding requested N_Omega, in counterclockwise direction
    #N_Omega is the number of discrete ordinances per quarter
    M = N_Omega * 4
    #ThetaX(m) = (2m+1)pi/M
    #mu(m) = cos(ThetaX(m))
    #eta(m) = sin(ThetaX(m))    
    Omega = []
    for iM in range(M):
        theta = (2*iM + 1)* np.pi / M
        mu = np.cos(theta)
        eta = np.sin(theta)
        Omega.append([mu, eta])
    return Omega
    
def ReturnQuadratureWeight(N_Omega):
    #returns the weight per quadrature point.
    #This problem uses equal weighting per direction
    M = N_Omega * 4
    Weight = []
    for iM in range(M):
        Weight.append(2 / M)
        #Weight.append(2*np.pi / M)
    return Weight

def Solve2DTransportDD(Space, N_CellsXY, N_Omega, BCs, ConvParam):
    #Solves 2D iterative transport with Diamond Difference scheme
    #The space is a series of cells linked to each other - west to east, south to north
    #N is the set of Nx and Ny cell counts, horizontally annd vertically
    Nx, Ny = N_CellsXY
    #Boundary Conditions (BCs) are the set of four boundary conditions for each edge: E, W, N, S
    #BC options are +1 (mirror) or -1 (vacuum)
    #N_Omega is the number of angles to iterate over per quadrant
    #ConvParam are the convergence paramaters - the maximum permissible difference between individual phi0s and the sum of the total
    ConvPhi, ConvPhiSum, iterMax = ConvParam
    #M is the total number of discrete ordinates
    M = N_Omega * 4
    #Reset QScat and PsiC for each cell
    for cell in Space:
        cell.ResetAngle(N_Omega)
    
    Converged = False
    Direction = ['NE', 'NW', 'SW', 'SE']
    #xStart    =  0  ,  Nx ,  Nx ,   0 
    #xStop     =  Nx ,  0  ,  0  ,  Nx
    #yStart    =  0  ,  0  ,  Ny ,  Ny
    #yStop     =  Ny ,  Ny ,  0  ,  0
    
    XStart = [0, Nx-1, Nx-1, 0   ]
    XStop  = [Nx,  -1,   -1, Nx  ]
    XLast  = [Nx-1, 0,    0, Nx-1]
    
    YStart = [0,    0, Ny-1, Ny-1]
    YStop  = [Ny,   Ny,  -1, -1  ]
    YLast  = [Ny-1, Ny-1, 0, 0   ]
    
    XStep  = [1, -1, -1, 1]
    YStep  = [1, 1, -1, -1]
    
    #l - prevX[quadrant] yields previous cell
    nextX  = [1, -1, -1, 1]
    nextY  = [Nx, Nx, -1*Nx, -1*Nx]
    EdgeX  = [1, 0, 0, 1] #(West, East, East, West)
    EdgeY  = [3, 3, 2, 2] #(South, South, North, North)

    #All fluxes at the boundaries are assumed zero initially, along with zero scattering source
    #This represents vacuum boundaries accurately, but not mirror boundaries, which will only be corrected with successive iterations
    Omega  = GenerateQuadrature(N_Omega)
    Weight = ReturnQuadratureWeight(N_Omega)
    PhiSumOld = 0.0
    iterCount = 0
    

    while not Converged:
        #Reset Scalar Flux
        for cell in Space:
            cell.Phi00 = 0.0
            cell.Phi10 = 0.0
            cell.Phi11 = 0.0
            #Reset Edge and Center Psis
            cell.PsiEdge[:] = 0.0,0.0,0.0,0.0
            #cell.PsiC = 0.0
        
        
        
        #l = i + Nx * j
        #the cell east is at l+1
        #the cell west is at l-1
        #the cell north is at l+Nx
        #the cell south is at l-Nx

        #1 - Solve for the angular fluxes throughout the space, for each angle in N_Omega
        # Use Diamond Differencing to solve for the next boundary
        

        #print("Current QAll[NE] map for iteration", iterCount)
        #for iY in range(Ny-1, -1, -1):
       #     for iX in range(Nx):
       #         l = iX + iY * Nx
        #        print(Space[l].QAll[0], end = " ")
        #    print("")
        
        for quad in range(4):
            print("iteration:",iterCount,". Direction:",Direction[quad])
            for omicron in range(N_Omega):
                mu, eta =  Omega[quad*N_Omega +  omicron]
                
                for iY in range(YStart[quad], YStop[quad], YStep[quad]):
                    for iX in range(XStart[quad], XStop[quad], XStep[quad]):
                        l = iX + iY * Nx
                        #print("l, iX, iY:",l, iX, iY)
                        DX = (Space[l].X1 - Space[l].X0) * XStep[quad]
                        DY = (Space[l].Y1 - Space[l].Y0) * YStep[quad]
                        Div = (Space[l].SigT + 2 * mu / DX + 2 * eta / DY) ** -1

                        #East-West Calculation
                        #Space[l].PsiC[quad*N_Omega +  omicron] =  Space[l].Psi(E/W) * A1 + Space[l].Psi(N/S) * A2 + Space[l].QAll[quad*N_Omega +  omicron] * A3 
                        Space[l].PsiC[quad*N_Omega + omicron] = (Space[l].PsiEdge[EdgeX[quad]] * 2 * mu / DX + Space[l].PsiEdge[EdgeY[quad]] * 2 * eta / DY + Space[l].QAll[quad*N_Omega + omicron]) * Div
                        #Next X direction Edge Psi 
                        if iX != XLast[quad]:
                            Space[l+nextX[quad]].PsiEdge[EdgeX[quad]] = 2 * Space[l].PsiC[quad*N_Omega + omicron] - Space[l].PsiEdge[EdgeX[quad]]
                            
                        if iY != YLast[quad]:
                            if l+nextY[quad] > len(Space):
                                print("out of bounds at: Space[",l,"] attempting to reach Space[",l+nextY[quad],"], with sweep direction:",Direction[quad])
                                break
                            
                            Space[l+nextY[quad]].PsiEdge[EdgeY[quad]] = 2 * Space[l].PsiC[quad*N_Omega + omicron] - Space[l].PsiEdge[EdgeY[quad]]
                            
                        
                        #East Mirror Corrections
                        #TODO: Add mirror boundary corrections
        
        
        #2 - Solve for the set of isotropic and linearly anisotropic scalar fluxes
        #Phi[l,m] = sum(n,OmegaN)[ w(omicro) * PsiC(omicro) * Y[l,m]  ]
                        Space[l].Phi00 += Weight[quad*N_Omega +  omicron] * Space[l].PsiC[quad*N_Omega + omicron] * (1) / 4
                        Space[l].Phi10 += Weight[quad*N_Omega +  omicron] * Space[l].PsiC[quad*N_Omega + omicron] * (mu * np.sqrt(3)) / 4
                        Space[l].Phi11 += Weight[quad*N_Omega +  omicron] * Space[l].PsiC[quad*N_Omega + omicron] * (-1*eta**2 * np.sqrt(3/2)) / 4

        #print("Current phi00 map for iteration", iterCount)
        #for iY in range(Ny-1, -1, -1):
        #    for iX in range(Nx):
       #         l = iX + iY * Nx
        #        print(Space[l].Phi00, end = " ")
        #    print("")
            
        

        #3 = Use the scalar fluxes from 2 to recalculate the scatter source terms in each cell, and ammend the total neutron fluxes
        #print("QAll calculations: l, Ordinate, Q")
        
        
        NYXO = Ny, Nx, N_Omega
        
        UpdateQAll(Space, NYXO, Omega)
        
        # for iY in range(Ny):
            # for iX in range(Nx):
                # l = iX + iY * Nx
                # for quad in range(4):
                    # for omicron in range(N_Omega):
                        # mu, eta =  Omega[quad*N_Omega + omicron]
                        # #if l == 14 or l == 15 or l == 20 or l == 21:
                        # #    print("Source Space SigS0, SigS1, Phi00, Qsrc")
                        # #    print(Space[l].SigS0, Space[l].SigS1, Space[l].Phi00, Space[l].Qsrc)
                        # Space[l].QAll[quad*N_Omega + omicron] = max(0.0, Space[l].Phi00) * Space[l].SigS0 + (max(0.0,Space[l].Phi10) * np.sqrt(3) * mu - 2 * max(0.0,Space[l].Phi11) * eta * eta * np.sqrt(3/2)) * Space[l].SigS1 + Space[l].Qsrc / N_Omega / 4
                        # #print(l,quad*N_Omega + omicron, Space[l].QAll[quad*N_Omega + omicron] )
                        

        
                
        #print("Phi calculations: l, Phi00, Phi10, Phi11:")
        #4 Convergence Testing
        phiMax = 0.0
        phiMin = 1.0
        Conv1 = True
        Conv2 = False
        PhiSum = 0.0
        ConvTest1 = 0.0
        for iY in range(Ny):
            for iX in range(Nx):
                l = iX + iY * Nx
                
                PhiTest = abs((Space[l].Phi00 - Space[l].PhiOld) / Space[l].Phi00)
                Space[l].PhiOld = Space[l].Phi00
                #print(l, Space[l].Phi00 , Space[l].Phi10 , Space[l].Phi11)
                phiMax = max(phiMax,Space[l].Phi00)
                phiMin = min(phiMin,Space[l].Phi00)
                #PhiTest = abs((Space[l].Phi00 + Space[l].Phi10 + Space[l].Phi11 - Space[l].PhiOld) / (Space[l].Phi00 + Space[l].Phi10 + Space[l].Phi11))
                ConvTest1 = max(PhiTest, ConvTest1)
                PhiSum += Space[l].PhiOld
                
                #Reset Phi00, 01, 11 values

        Conv1 = ConvTest1 <  ConvPhi 
        ConvTest2 = abs((PhiSum - PhiSumOld) / max(PhiSum, 1.0E-8))
        print("conv1 and conv2 tests for iteration",iterCount,": ",ConvTest1, ConvTest2)    
        print("phiMax and phiMin for this iteration:",phiMax, phiMin) 
        Conv2 = ConvTest2 < ConvPhiSum
        PhiSumOld = PhiSum
        Converged = Conv1 and Conv2
        
        
        if iterCount >= iterMax:
            print("iterations have exceeded max at", iterCount,"\nCurrent ")
            break
        iterCount += 1

def UpdateQAll(Space, N, Omega):
    
    Ny, Nx, N_Omega = N
    for iY in range(Ny):
        for iX in range(Nx):
            l = iX + iY * Nx
            for quad in range(4):
                for omicron in range(N_Omega):
                    mu, eta =  Omega[quad*N_Omega + omicron]
                    #if l == 14 or l == 15 or l == 20 or l == 21:
                    #    print("Source Space SigS0, SigS1, Phi00, Qsrc")
                    #    print(Space[l].SigS0, Space[l].SigS1, Space[l].Phi00, Space[l].Qsrc)

                    Space[l].QAll[quad*N_Omega + omicron] = max(0.0, Space[l].Phi00) * Space[l].SigS0 + (max(0.0,Space[l].Phi10) * np.sqrt(3) * mu - 2 * max(0.0,Space[l].Phi11) * eta * eta * np.sqrt(3/2)) * Space[l].SigS1 + Space[l].Qsrc / N_Omega / 4
    
def Solve2DCellFlux(cell, Quadrant, Omega, type):
    #Solves the angular flux within a cell for both Diamond Difference (type = "DD" or 1) and Step Characteristics (type = "SC" or 2)
    #Omega consists of three parts: mu, eta, and omicron
    mu, eta, omicron = Omega
    #Edge - 0 East, 1 West, 2 North, 3 South

    PsiGX   = cell.PsiEdge[GivenX[Quadrant]]
    PsiGY   = cell.PsiEdge[GivenY[Quadrant]]    
    Q       = cell.QAll[omicron]
    DX      = cell.X1 - cell.X0
    DY      = cell.Y1 - cell.Y0
    SigT    = cell.SigT
    
    if type == "DD" or type == 1:   #Diamond Difference Scheme
        PsiC  = (PsiGX * 2 * mu / DX + PsiGY * 2 * eta / DY + Q) / (SigT+ 2 * mu / DX + 2 * eta / DY)
        PsiSX = 2*PsiC - PsiGX
        PsiSY = 2*PsiC - PsiGY
    
    elif type == "SC" or type == 2:  #Step Characteristics
        Tau = SigT * DY / eta
        Exp = 1-np.exp(-1*Tau)
        DXp = mu * Tau / SigT
        Ax  = DXp / DX
        Q = Q/SigT
        
        PsiSY = Ax * (Q + (PsiGX - Q) * Exp / Tau) + (1-Ax) * (PsiGY * np.exp(-1*Tau) + Q * Exp)
        PsiSX = Q + Exp * (PsiGY - Q) / Tau
        PsiC  = Q - mu / SigT / DX * (cell.PsiEdge[0] - cell.PsiEdge[1]) - eta / SigT / DY * (cell.PsiEdge[2] - cell.PsiEdge[3])
                        
    return PsiC, PsiSX, PsiSY
    
        

def Calc1DTransport(PsiIn, Tau, QT):
    return PsiIn * np.exp(-1*Tau) + QT * (1-np.exp(-1*Tau))

def Calc2DComponentsA(PsiGX, PsiGY, Tau, QT, DY, DYp):
    Ay = DYp/DY
    
    PsiGY1 = Calc1DTransport(PsiGY, Tau, QT)
    PsiGX1 = Calc1DTransport(PsiGX, Tau, QT)
    
    PsiSY = QT + (PsiGX - PsiGX1) / Tau
    PsiSX = Ay * (QT + (PsiGY - PsiGY1) / Tau) + (1-Ay) * PsiGX1
    
    return PsiSY, PsiSX
    
def Calc2DComponentsB(PsiGX, PsiGY, Tau, QT, DX, DXp):
    Ax = DXp/DX
    
    PsiGY1 = Calc1DTransport(PsiGY, Tau, QT)
    PsiGX1 = Calc1DTransport(PsiGX, Tau, QT)
    
    PsiSX = QT + (PsiGY - PsiGY1) / Tau
    PsiSY = Ax * (QT + (PsiGX - PsiGX1) / Tau) + (1-Ax) * PsiGY1
    
    return PsiSY, PsiSX
        
        
def Solve2DTransportSC(Space, N_CellsXY, N_Omega, BCs, ConvParam):
    #Solves 2D iterative transport via step characteristics
    #The space is a series of cells linked to each other - west to east, south to north
    #N is the set of Nx and Ny cell counts, horizontally annd vertically
    Nx, Ny = N_CellsXY
    #Boundary Conditions (BCs) are the set of four boundary conditions for each edge: E, W, N, S
    #BC options are +1 (mirror) or -1 (vacuum)
    #N_Omega is the number of angles to iterate over per quadrant
    #ConvParam are the convergence paramaters - the maximum permissible difference between individual phi0s and the sum of the total
    ConvPhi, ConvPhiSum, iterMax = ConvParam
    #Reset Q and PsiC for each cell
    for cell in Space:
        cell.ResetAngle(N_Omega)
    
    Converged = False
    Direction = ['NE', 'NW', 'SW', 'SE']
    #xStart    =  0  ,  Nx ,  Nx ,   0 
    #xStop     =  Nx ,  0  ,  0  ,  Nx
    #yStart    =  0  ,  0  ,  Ny ,  Ny
    #yStop     =  Ny ,  Ny ,  0  ,  0
    
    XStart = [0, Nx-1, Nx-1, 0   ]
    XStop  = [Nx,  -1,   -1, Nx  ]
    XLast  = [Nx-1, 0,    0, Nx-1]
    
    YStart = [0,    0, Ny-1, Ny-1]
    YStop  = [Ny,   Ny,  -1, -1  ]
    YLast  = [Ny-1, Ny-1, 0, 0   ]
    
    XStep  = [1, -1, -1, 1]
    YStep  = [1, 1, -1, -1]
    
    #l - prevX[quadrant] yields previous cell
    nextX  = [1, -1, -1, 1]
    nextY  = [Nx, Nx, -1*Nx, -1*Nx]
    SolveX = [0, 1, 1, 0]
    SolveY = [2, 2, 3, 3]
    GivenX  = [1, 0, 0, 1] #(West, East, East, West)
    GivenY  = [3, 3, 2, 2] #(South, South, North, North)

    #All fluxes at the boundaries are assumed zero initially, along with zero scattering source
    #This represents vacuum boundaries accurately, but not mirror boundaries, which will only be corrected with successive iterations
    Omega  = GenerateQuadrature(N_Omega) #Only need the NE Angles, as the rest are repeated
    Weight = ReturnQuadratureWeight(N_Omega)
    PhiSumOld = 0.0
    iterCount = 0
    

    while not Converged:
        #Reset Scalar Flux
        for cell in Space:
            cell.Phi00 = 0.0
            cell.Phi10 = 0.0
            cell.Phi11 = 0.0
            #Reset Edge and Center Psis
            
            #cell.PsiC = 0
        #l = i + Nx * j
        #the cell east is at l+1
        #the cell west is at l-1
        #the cell north is at l+Nx
        #the cell south is at l-Nx

        #1 - Solve for the angular fluxes throughout the space, for each angle in N_Omega
        # Use Step characteristics to solve for the next boundary
        

        #print("Current QAll[NE] map for iteration", iterCount)
        #for iY in range(Ny-1, -1, -1):
       #     for iX in range(Nx):
       #         l = iX + iY * Nx
        #        print(Space[l].QAll[0], end = " ")
        #    print("")
        
        for quad in range(4):
            #print("iteration:",iterCount,". Direction:",Direction[quad])
            #print("l, mu, PsiC")
            for omicron in range(N_Omega):
                mu, eta =  Omega[omicron]
                for cell in Space:
                    cell.PsiEdge[:] = 0.0,0.0,0.0,0.0
                
                
                for iY in range(YStart[quad], YStop[quad], YStep[quad]):
                    for iX in range(XStart[quad], XStop[quad], XStep[quad]):
                        l = iX + iY * Nx
                        #Space[l].PsiEdge[:] = 0.0,0.0,0.0,0.0
                        #print("l, iX, iY:",l, iX, iY)
                        DX = (Space[l].X1 - Space[l].X0)
                        DY = (Space[l].Y1 - Space[l].Y0)
                        SigT = Space[l].SigT
                        QT = Space[l].QAll[quad*N_Omega + omicron] / SigT
                        Tau = 0
                        DYp = DX * eta / mu
                        
                        PsiGX = Space[l].PsiEdge[GivenX[quad]]
                        PsiGY = Space[l].PsiEdge[GivenY[quad]]
                        
                        if DYp >= 0:    #Type A
                            Tau = SigT * DX / mu
                            PsiSY, PsiSX = Calc2DComponentsA(PsiGX, PsiGY, Tau, QT, DY, DYp)
                            
                        
                        
                        else:           #Type B
                            DXp = DY * mu / eta
                            Tau = SigT * DY / eta
                            PsiSY, PsiSX = Calc2DComponentsB(PsiGX, PsiGY, Tau, QT, DX, DXp)
                        
                        
                        Space[l].PsiEdge[SolveX[quad]] = PsiSX
                        Space[l].PsiEdge[SolveY[quad]] = PsiSY
                        Space[l].PsiC[quad*N_Omega + omicron] = QT - mu / SigT / DX * (PsiSX - PsiGX) - eta / SigT / DY * (PsiSY - PsiGY)
                        #print(l, mu, Space[l].PsiC[quad*N_Omega + omicron])
                        #Next X direction Edge Psi 
                        if iX != XLast[quad]:
                            Space[l+nextX[quad]].PsiEdge[GivenX[quad]] = Space[l].PsiEdge[SolveX[quad]]
                            
                        if iY != YLast[quad]:
                            if l+nextY[quad] > len(Space):
                                print("out of bounds at: Space[",l,"] attempting to reach Space[",l+nextY[quad],"], with sweep direction:",Direction[quad])
                                break
                            
                            Space[l+nextY[quad]].PsiEdge[GivenY[quad]] = Space[l].PsiEdge[SolveY[quad]]
                            
                        
                        #East Mirror Corrections
                        #TODO: Add mirror boundary corrections
        
        
        #2 - Solve for the set of isotropic and linearly anisotropic scalar fluxes
        #Phi[l,m] = sum(n,OmegaN)[ w(omicro) * PsiC(omicro) * Y[l,m]  ]
                        Space[l].Phi00 += Weight[quad*N_Omega +  omicron] * Space[l].PsiC[quad*N_Omega + omicron] * (1) / 4
                        Space[l].Phi10 += Weight[quad*N_Omega +  omicron] * Space[l].PsiC[quad*N_Omega + omicron] * (mu * np.sqrt(3)) / 4
                        Space[l].Phi11 += Weight[quad*N_Omega +  omicron] * Space[l].PsiC[quad*N_Omega + omicron] * (-1*eta**2 * np.sqrt(3/2)) / 4
                        
                        #print(l, mu, Space[l].PsiC[quad*N_Omega + omicron])

        #print("Current phi00 map for iteration", iterCount)
        #for iY in range(Ny-1, -1, -1):
        #    for iX in range(Nx):
       #         l = iX + iY * Nx
        #        print(Space[l].Phi00, end = " ")
        #    print("")
            
        NYXO = Ny, Nx, N_Omega
        
        UpdateQAll(Space, NYXO, Omega)

        # #3 = Use the scalar fluxes from 2 to recalculate the scatter source terms in each cell, and ammend the total neutron fluxes
        # #print("QAll calculations: l, Ordinate, Q")
        # for iY in range(Ny):
            # for iX in range(Nx):
                # l = iX + iY * Nx
                # for quad in range(4):
                    # for omicron in range(N_Omega):
                        # mu, eta =  Omega[quad*N_Omega + omicron]
                        # #if l == 14 or l == 15 or l == 20 or l == 21:
                        # #    print("Source Space SigS0, SigS1, Phi00, Qsrc")
                        # #    print(Space[l].SigS0, Space[l].SigS1, Space[l].Phi00, Space[l].Qsrc)
                        # Space[l].QAll[quad*N_Omega + omicron] = max(0.0, Space[l].Phi00) * Space[l].SigS0 + (max(0.0,Space[l].Phi10) * np.sqrt(3) * mu - 2 * max(0.0,Space[l].Phi11) * eta * eta * np.sqrt(3/2)) * Space[l].SigS1 + Space[l].Qsrc / N_Omega / 4
                        # #print(l,quad*N_Omega + omicron, Space[l].QAll[quad*N_Omega + omicron] )
                        

        
                
        #print("Phi calculations: l, Phi00, Phi10, Phi11:")
        #4 Convergence Testing
        phiMax = 0.0
        phiMin = 1.0
        Conv1 = True
        Conv2 = False
        PhiSum = 0.0
        ConvTest1 = 0.0
        for iY in range(Ny):
            for iX in range(Nx):
                l = iX + iY * Nx
                
                PhiTest = abs((Space[l].Phi00 - Space[l].PhiOld) / Space[l].Phi00)
                Space[l].PhiOld = Space[l].Phi00
                #print(l, Space[l].Phi00 , Space[l].Phi10 , Space[l].Phi11)
                phiMax = max(phiMax,Space[l].Phi00)
                phiMin = min(phiMin,Space[l].Phi00)
                #PhiTest = abs((Space[l].Phi00 + Space[l].Phi10 + Space[l].Phi11 - Space[l].PhiOld) / (Space[l].Phi00 + Space[l].Phi10 + Space[l].Phi11))
                ConvTest1 = max(PhiTest, ConvTest1)
                PhiSum += Space[l].PhiOld
                

        Conv1 = ConvTest1 <  ConvPhi 
        ConvTest2 = abs((PhiSum - PhiSumOld) / max(PhiSum, 1.0E-8))
        print("conv1 and conv2 tests for iteration",iterCount,": ",ConvTest1, ConvTest2)    
        print("phiMax and phiMin for this iteration:",phiMax, phiMin) 
        Conv2 = ConvTest2 < ConvPhiSum
        PhiSumOld = PhiSum
        Converged = Conv1 and Conv2
        
        if iterCount >= iterMax:
            print("iterations have exceeded max at", iterCount,"\nCurrent ")
            break
        iterCount += 1

def Create2DResults(Space, Lines, N2D):
    #This assumes that the Space has already had its flux solved for. This will create a 2D matrix of scalar flux, along with scatter plot (Phi(R) data along the given lines
    #Lines consists of sets of 4 values: XStart, YStart, XStop, YStart
    m = []
    b = []
    PhiXSet = []
    PhiYSet = []
    LCount = 0
    Xvector = []
    Yvector = []
    for line in Lines:
        X0, Y0, X1, Y1 = line
        #F(x) = mx +b
        m.append((Y1 - Y0)/(X1 - X0))
        b.append(Y1 - (Y1 - Y0)/(X1 - X0) * X1)
        PhiXSet.append([])#Adds another dimension of data
        PhiYSet.append([])#Adds another dimension of data
        LCount += 1
    #N2D is the number of cells in the X and Y directions, respectively        
    Nx, Ny = N2D

    PhiMatrix = np.zeros([Nx, Ny])
    l = 0
    for iY in range(Ny):
        
        for iX in range(Nx):
        
            l = iX + iY * Nx
            #print("l,iX, iY:",l,iX, iY)
            #print(Space[l].Phi00)
            PhiMatrix[iX, iY] = max(Space[l].Phi00, 1e-10)
            
            CX0, CX1, CY0, CY1, CZ0, CZ1 = Space[l].GEO()
            #print("CX0, CX1, CY0, CY1:",Space[l].GEO()[0:4])
            
            if iY == 0:
                Xvector.append(CX0 + (CX1 - CX0)/2)
            
            for im in range(LCount):
                LY0 = m[im]*CX0 + b[im]
                LY1 = m[im]*CX1 + b[im]
                case1 = LY0>=CY0 and LY0<=CY1
                case2 = LY1>=CY0 and LY1<=CY1
                case3 = LY0<=CY0 and LY1>=CY1
                
                #print("LY0, LY1:",LY0, LY1)
                
                if case1 or case2 or case3:
                    if case1 and case2:
                        LX0 = CX0
                        LX1 = CX1
                    elif case1:
                        LX0 = CX0
                        LX1 = (CY1 - b[im]) / m[im]
                    elif case2:
                        LX1 = CX1
                        LX0 = (CY0 - b[im]) / m[im]
                    elif case3:
                        LX0 = (CY0 - b[im]) / m[im]
                        LX1 = (CY1 - b[im]) / m[im]
                    R0 = np.sqrt(LX0**2 + LY0**2)
                    R1 = np.sqrt(LX1**2 + LY1**2)
                    R = R0 + (R1-R0)/2

                    PhiXSet[im].append(R)
                    PhiYSet[im].append(max(1e-8,Space[l].Phi00))
        Yvector.append(Space[l].Y0 + (Space[l].Y1 - Space[l].Y0)/2)
    return PhiMatrix, Xvector, Yvector, PhiXSet, PhiYSet, OuterAvg(Space)    
            
def TransportSolve(Space, N_CellsXY, N_Omega, BCs, ConvParam, Lines, type):
    #Takes in the inputs for either Transport Solver and returns the 2D results tuple.
    #Type has a value of 1 or DD or Diamond for diamond difference and 2 or SC or Step for Step Characteristics
    if type == 1 or type == "DD" or type == "Diamond":
        Solve2DTransportDD(Space, N_CellsXY, N_Omega, BCs, ConvParam)
    elif type == 2 or type == "SC" or type == "Step":
        Solve2DTransportSC(Space, N_CellsXY, N_Omega, BCs, ConvParam)
    return    Create2DResults(Space, Lines, N_CellsXY)    
            
#Analysis
#Each material gets a data set
#For each data set, sensitivity study at n = 2, 4, 8 (total N 216, 12^3, 24^)
#For each sensitivity study, display the long diagonal and the bisection
#For each plot, display the average outside flux

#1 - Select Material and dimensions
# SetChoice = 1
# SetW        = 10 ** (SetChoice - 1)
#SetW = 2
# MatChoice = 'Air'                                     
# MatName = "Air"
# NPoints = 100                  

# LongStart  = [0,0,0]
# LongStop   = [SetW,SetW,SetW]
# ShortStart = [0, SetW/2, SetW/2]
# ShortStop  = [SetW, SetW/2, SetW/2]
# BCList3D = [-1, -1, -1, -1, -1, -1]

BCs = [-1,-1,-1,-1]

# Lines = []
# Lines.append([0, 0, SetW, SetW])
# Lines.append([0, SetW/2, SetW, SetW/2]) 

# InnerN  = 4
# BarrierN  = 3
# OuterN = InnerN + BarrierN*2
# N_CellsXY = OuterN,OuterN

N_Omega = 10
ConvParam = [0.0001, 0.0001, 100]

#Square = GenerateSquare(MatChoice, SetChoice, InnerN, BarrierN)
#SquareSpace = Create2DSpace(Square)

# Solve2DTransportDD(SquareSpace, N_CellsXY, N_Omega, BCs, ConvParam)


#For Each method (2)

MethodName = "Diamond Difference", "Step Characteristics"

#For each material/size set (4) iMat
MatSet = "Ref", "Sca", "Abs", "Air"
MatName = "Reflector", "Scatterer", "Absorber", "Air"
SpaceSet = 1,2,0,3

#SetW        = 10 ** (SpaceSet[iMat] - 1)

#For each of three Angular discretization (1,4,16 discrete ordinates)
#N_OmegaSet = 1, 4, 16# = 4^iOmega
#N_Omega = N_OmegaSet[iOmega]

#Create one figure
#3x3 Subplots - Diagonal Display, Horizontal Display, Heat Map ; N = 100, N = 400, N = 10000
#NSide = 10, 20, 100 - InnerN = 4, 8, 40;   BarrierN = 3, 6, 30
NSide = 1, 4, 16
iFig = 1
for iMethod in range(2):#2
    
    for iMaterial in range(4):#4
        MatChoice = MatSet[iMaterial]
        SetChoice = SpaceSet[iMaterial]
        SetW = 10 ** (SpaceSet[iMaterial] - 1)
        Lines = []
        Lines.append([0, 0, SetW, SetW])
        Lines.append([0, SetW/2, SetW, SetW/2]) 
        
        for iOmega in range(3):#3
            N_Omega = 4 ** iOmega
            fig = plt.figure(iFig)
            fig.suptitle("Figure "+str(iFig)+": "+MethodName[iMethod]+", "+MatName[iMaterial]+" with "+str(N_Omega)+" Discrete Ordinates", fontsize = 20)
            
            plt.subplots_adjust(top=0.92, bottom=0.075, left=0.06, right=0.975, hspace=0.305, wspace=0.215)
            
            for iN in range(3):

                InnerN = NSide[iN] * 4
                BarrierN = NSide[iN] * 3
                NLength = InnerN + 2 * BarrierN
                N_CellsXY = NLength, NLength
                NTotal = NLength ** 2

                Square = GenerateSquare(MatChoice, SetChoice, InnerN, BarrierN)
                SquareSpace = Create2DSpace(Square)
                PhiMatrix, Xvector, Yvector, PhiXSet, PhiYSet, OuterAvg1 = TransportSolve(SquareSpace, N_CellsXY, N_Omega, BCs, ConvParam, Lines, iMethod+1)
                
                X_Diag, X_Flat = PhiXSet
                P_Diag, P_Flat = PhiYSet
                #Diagonal Plot
                x1 = X_Diag
                y1 = np.log(P_Diag)
                y2 = np.log([OuterAvg1, OuterAvg1])
                x2 = [x1[0],x1[-1]]
                
                plt.subplot(3,3,3*iN+1)
                plt.plot(x1,y1, label="Phi")
                plt.plot(x2,y2, label="Phi_Avg")
                plt.xlabel("X [cm]")
                plt.ylabel("Log Flux [n/cm^2/sec]") 
                plt.title("Diagonal Flux with N = "+str(NTotal)) 
                plt.legend()
                                
                
                #Horizontal Plot
                x1 = X_Flat
                y1 = np.log(P_Flat)
                y2 = np.log([OuterAvg1, OuterAvg1])
                x2 = [x1[0],x1[-1]]
                
                plt.subplot(3,3,3*iN+2)
                plt.plot(x1,y1, label="Phi")
                plt.plot(x2,y2, label="Phi_Avg")
                plt.xlabel("X [cm]")
                plt.ylabel("Log Flux [n/cm^2/sec]") 
                plt.title("Diagonal Flux with N = "+str(NTotal)) 
                plt.legend()
                
                
                #Heat Map
                ax = plt.subplot(3,3,3*iN+3)
                #ax.set_title('Fluxmap')
                plt.pcolormesh(Xvector, Yvector, PhiMatrix)
                plt.title('Fluxmap for N ='+str(NTotal))
                plt.colorbar()
            iFig+=1







plt.show()


# PlotPhi = []
# PlotX   = []
# phiAvg  = []
# PlotHeat = []
# XVec = []
# YVec = []

# NDisplay = []




# #print("Space Test:\nCell#  X  Y  Q")
# #l = 0
# #for cell in SquareSpace:
# #    X = cell.X0 + (cell.X1-cell.X0)/2
# #    Y = cell.Y0 + (cell.Y1-cell.Y0)/2
# #    Q = cell.Qsrc
# #    print(l, X, Y, Q)
# #    l+=1


# BCs = [-1,-1,-1,-1]
# ConvParam = [0.1, 0.1, 3]
# N_Omega = 4

# print(GenerateQuadrature(N_Omega))
# print(ReturnQuadratureWeight(N_Omega))





# #2 - Loop through N values
# for iN in range(3):
    # InnerN  = 5**(iN+1)
    # BarrierN  = 5*(iN+1)
    


    # OuterN = InnerN + BarrierN*2
    # N_CellsXY = OuterN,OuterN
    # NDisplay.append(N_CellsXY[0]*N_CellsXY[1])
    
    
# #3 - For each N value, store Long and Short path, along with the average outside flux
    # Square = GenerateSquare(MatChoice, SetChoice, InnerN, BarrierN)
    # SquareSpace = Create2DSpace(Square)
    # Solve2DTransportDD(SquareSpace, N_CellsXY, N_Omega, BCs, ConvParam)
    # PhiMat, Xvector, Yvector, Xset, Yset, PhiBar = Create2DResults(SquareSpace, Lines, N_CellsXY)
    
    # print("X:",Xvector)
    # print("Y:",Yvector)
    # print("Map:\n",PhiMat)
    
    
    # phiAvg.append(PhiBar)
    # #Flux Map
    # PlotHeat.append(PhiMat)
    # XVec.append(Xvector)
    # YVec.append(Yvector)
    # #Long Results
    # PhiLong, XLong = Yset[1], Xset[1]
    # PlotPhi.append(PhiLong)
    # PlotX.append(XLong)
    # #Short Results
    # PhiShort, XShort = Yset[0], Xset[0]
    # PlotPhi.append(PhiShort)
    # PlotX.append(XShort)
    
# #4 - Create 3x3 plot, each of which has the AVG, Long diagonal on the left and short path in the middle and heat map on the right

# for iPlt in range(3):
    # iPlot = iPlt*3
    # x1 = PlotX[iPlot]
    # y1 = np.log(PlotPhi[iPlot])
    # y2 = np.log([phiAvg[iPlt], phiAvg[iPlt]])
    # x2 = [x1[0],x1[-1]]
    
    # plt.subplot(3,3,iPlot+1)
    # plt.plot(x1,y1, label="Phi")
    # plt.plot(x2,y2, label="Phi_Avg")
    # plt.xlabel("X [cm]")
    # plt.ylabel("Log Flux [n/cm^2/sec]") 
    # plt.title("Diagonal Flux with N = "+str(NDisplay[iPlt]))  
    # plt.legend()
    
    # iPlot = iPlt*3+1
    # x1 = PlotX[iPlot]
    # y1 = np.log(PlotPhi[iPlot])
    # y2 = np.log([phiAvg[iPlt], phiAvg[iPlt]])
    # x2 = [x1[0],x1[-1]]
    
    # plt.subplot(3,3,iPlot+1)
    # plt.plot(x1,y1, label="Phi")
    # plt.plot(x2,y2, label="Phi_Avg")
    # plt.xlabel("X [cm]")
    # plt.ylabel("Log Flux [n/cm^2/sec]") 
    # plt.title("Horizontal Flux with N = "+str(NDisplay[iPlt]))
    # plt.legend()

    # iPlot = iPlt*3+2
    
    # # #Start with some Phi(x,y)
    # x1, x2 = np.meshgrid(XVec[iPlt], YVec[iPlt], indexing='ij')
    # plt.subplot(3,3,iPlot+1)
    # plt.pcolormesh(x1, x2, PlotHeat[iPlt], shading='auto')
    # plt.colorbar()

    # plt.xlabel("X[cm]")
    # plt.ylabel("Y[cm]")
    # #Create color legend
    # #Create Title string with significant values
    # ax.set_title(Title_String)
    # fig.tight_layout()



    
# plt.suptitle(MatName+" Transport Plots", fontsize = 20)
# 






# plt.show()
