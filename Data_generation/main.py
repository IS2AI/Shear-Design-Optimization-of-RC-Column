import csv
import os
from functions import *
import column
import math
import numpy as np

# Fixed parameters, need to be defined at the beginning
Height = 3.5    # [m] Column height
fck = 25        # [MPa] Characteristic compressive strength
V_ref = 5.5
# Creating a dictionary to store the design parameters
# data = empty_data_for_params()
data = {'P': [], 'My': [], 'Mz': [], 'Vy': [], 'Vz': [], 'fck': [], 'h': [], 'Width': [], 'Depth': [], 'D_rebar': [],
        'w_g': [], 'd_g': [], 'numRebars': [], 'As_total': [], 'D_tr': [], 's_tr': [], 'A_sw_y': [], 'A_sw_z': [],
        'Volume_tr': [], 'N_tr': []}

for z in range(1):
    print("First loop: range = 1")
    n = 10      # number of column designs
    numSaveToFile = 1   # number of designs to save

    # Creating a file to store the opensees logs and cash
    logName = './logs_' + str(z) + '.log'
    # Creating a csv file to store the generated dataset
    fileName = './data_' + str(z) + '.csv'
    # Create an object of class Columns to call the material, geometry and analysis parameters
    parameters = column.Column()
    # Saving all logs in the file, instead of terminal
    ops.logFile(logName, '-noEcho')

    with open(fileName, 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(
            ['P', 'My', 'Mz', 'Vy', 'Vz',  'fck', 'h', 'Width', 'Depth', 'D_rebar', 'w_g', 'd_g', 'numRebars',
             'As_total', 'D_tr', 's_tr', 'A_sw_y', 'A_sw_z', 'Volume_tr', 'N_tr'])

    i = 1
    # ***** START: Outer loop for iterating over different column designs *****
    while i < n + 1:
        print("Second loop: different column designs")
        # Concrete parameters
        fcd = -fck/parameters.g_c     # Design compressive strength
        eps1U = parameters.eps1U
        eps2U = parameters.eps2U

        # Steel parameters
        fyd = parameters.fyk/parameters.g_s             # Design yield strength of longitudinal rebars
        fyd_tr = parameters.fyk_tr / parameters.g_s     # Design yield strength of transverse rebars

        # Randomly generating column cross section
        Width, Depth = cross_section(Height)

        # Selecting rebar diameter
        D_rebar = random.choice(parameters.d_rebars)

        # Calculating area of one rebar
        As_bar = math.pi * D_rebar ** 2 / 4

        # Selecting the transverse rebar diameter
        D_min = max(D_rebar / 4, 0.006)                 # Eurocode page 162 section 9.5.3
        a = [x - D_min for x in parameters.d_rebars]
        D_index_min = a.index(min([i_min for i_min in a if i_min>0]))
        # D_tr = parameters.d_rebars_tr[D_index]  # Transverse rebar diameter

        # Max tie diameter is assumed to be D_rebar
        D_max = D_rebar
        D_index_max = parameters.d_rebars_tr.tolist().index(D_max)
      
        for diff_d_tr in range(D_index_max-D_index_min):
            D_tr = parameters.d_rebars_tr[D_index_min+diff_d_tr]
      

            # Calculating steel area and reinforcement constraints
            A_core = Width * Depth                      # Section gross area
            As_min = 0.002 * A_core                     # Min reinforcement area
            As_max = 0.04 * A_core                      # Max reinforcement area
            numRebars_min = math.ceil(As_min/As_bar)    # Min number of rebars
            numRebars_max = math.floor(As_max/As_bar)   # Max number of rebars

            # Total number of longitudinal-reinforcement bars in steel layer (symmetric top & bot)
            try:
                numBarsSec = random.randint(numRebars_min, numRebars_max)
                if numBarsSec<4:
                 
                    continue
            except:
                continue

            # Geometry modelling parameters
            coverY = Depth/2.0                          # Distance from the section z-axis to the edge of the cover concrete -- outer edge of cover concrete
            coverZ = Width/2.0                          # Distance from the section y-axis to the edge of the cover concrete -- outer edge of cover concrete
            coreY = coverY-parameters.cover-D_rebar/2-D_tr   # Distance from the section z-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
            coreZ = coverZ-parameters.cover-D_rebar/2-D_tr   # Distance from the section y-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
            dist1 = coverY-parameters.cover/2
            dist2 = coverZ-parameters.cover/2

            # Generating the grid parameters for rebars placement
            listDivisorPairs = returnDivisorsPair(numBarsSec)
            if len(listDivisorPairs)==1:
                listDivisorPairs = returnDivisorsPair(numBarsSec-1)

            w_g, d_g = grid_params(listDivisorPairs)
            w_h = (Width-2*D_rebar-2.5*parameters.cover-2*D_tr)/Width
            d_h = (Depth-2*D_rebar-2.5*parameters.cover-2*D_tr)/Depth

            rebarZ = np.linspace(-coreZ, coreZ, int(w_g))    # Coordinates of rebars in Z-axis
            rebarY = np.linspace(-coreY, coreY, int(d_g))    # Coordinates of rebars in Y-axis
            spacingZ = 2*coreZ/(w_g-1)                  # Spacing b/w rebars in Z-axis
            spacingY = 2*coreY/(d_g-1)                  # Spacing b/w rebars in Y-axis

            # Checking for minimal spacing requirement
            spacing_min = max(2*D_rebar, D_rebar+0.032+0.005, D_rebar+0.020)  # [m]
            if spacingZ<spacing_min or spacingY<spacing_min:
                continue

            # Cleaning the cash and saved parameters from previous design
            ops.wipe()

            # Defining model builder
            ops.model('basic', '-ndm', 3, '-ndf', 6)

            # Creating concrete material
            ops.uniaxialMaterial('Concrete01', parameters.IDcon, fcd, eps1U, fcd, eps2U)
            # Creating steel material
            ops.uniaxialMaterial('Steel01', parameters.IDreinf, fyd, parameters.Es, parameters.Bs)

            # Creating section
            ops.section('Fiber', parameters.SecTag, '-GJ', 1.0e6)
            # Creating the concrete core fibers
            ops.patch('quadr', parameters.IDcon, parameters.num_fib, parameters.num_fib, -coreY, coreZ, -coreY, -coreZ,
                      coreY, -coreZ, coreY, coreZ)
            # Creating the concrete cover fibers (top, bottom, left, right)
            ops.patch('quadr', parameters.IDcon, 1, parameters.num_fib, -coverY, coverZ, -coreY, coreZ, coreY, coreZ,
                      coverY, coverZ)
            ops.patch('quadr', parameters.IDcon, 1, parameters.num_fib, -coreY, -coreZ, -coverY, -coverZ, coverY, -coverZ,
                      coreY, -coreZ)
            ops.patch('quadr', parameters.IDcon, parameters.num_fib, 1, -coverY, coverZ, -coverY, -coverZ, -coreY, -coreZ,
                      -coreY, coreZ)
            ops.patch('quadr', parameters.IDcon, parameters.num_fib, 1, coreY, coreZ, coreY, -coreZ, coverY, -coverZ,
                      coverY, coverZ)

            # Inserting rebars along the section perimeter
            hollowY = d_h*coverY
            hollowZ = w_h*coverZ
            rebars_YZ = np.empty((0, 2))

            for ii, Y in enumerate(rebarY):
                for jj, Z in enumerate(rebarZ):
                    if abs(Y)<hollowY and abs(Z)<hollowZ:
                        continue
                    rebars_YZ = np.vstack([rebars_YZ, [Y, Z]])
            for ii in range(len(rebars_YZ)):
                ops.fiber(*rebars_YZ[ii], As_bar, parameters.IDreinf)

            # Checking for number of rebars in final configuration
            numTotRebars = len(rebars_YZ)
            if numTotRebars<4 or numTotRebars*As_bar<As_min:
                continue

            eps = fyd / parameters.Es   # Steel yield strain
            d_y = Width-parameters.cover-D_rebar/2-D_tr  # Distance from column outer edge to rebar in Y-axis
            d_z = Depth-parameters.cover-D_rebar/2-D_tr   # Distance from column outer edge to rebar in Z-axis
            Ky = eps/(0.7*d_y)      # Yield curvature in Y-axis
            Kz = eps/(0.7*d_z)      # Yield curvature in Z-axis

            As = As_bar*numTotRebars    # Total reinforcement area
            Ac = Width*Depth-As         # Total concrete area

            if d_g == 2:
                N_arr_Z = 2
            elif spacingY > 0.150 and d_g >= 3:
                N_arr_Z = d_g
            elif spacingY > 0.075 and d_g >= 3:
                N_arr_Z = math.ceil((d_g-1)/3)+1
            else:
                N_arr_Z = math.ceil((d_g-1)/5)+1

            if w_g == 2:
                N_arr_Y = 2
            elif spacingZ > 0.150 and w_g >= 3:
                N_arr_Y = w_g
            elif spacingZ > 0.075 and w_g >= 3:
                N_arr_Y = math.ceil((w_g-1)/3)+1
            else:
                N_arr_Y = math.ceil((w_g-1)/5)+1

            alpha_param = 1     # For non prestressed concrete
            v1 = 0.6 * (1 - fck / 250)
            cotTheta_assumed = 1  # Assumed to find the minimum spacing

            # Spacing range b/w transverse rebars
            s_min = 0.05 #round((min(N_arr_Y, N_arr_Z)*(math.pi*D_tr**2/4)*Height*fyd_tr*(1+cotTheta_assumed**2)/(alpha_param*v1*(-fcd)*min(Width, Depth)))**(1/2), 3)
            s_max = min(20 * D_rebar, Depth, Width, 0.4)    # Max spacing (see Eurocode 2 page 163)
            if s_min>s_max:
                continue

            spacingArray = int(round(s_min/0.05))
            spacing_tr = round(0.05 * spacingArray, 2) 
            
            
            N_tr = math.floor(Height / spacing_tr)  # Number of ties in a column

            # Cross-sectional area of the shear reinforcement
            A_sw_y = (math.pi * D_tr ** 2 / 4)*N_arr_Y
            A_sw_z = (math.pi * D_tr ** 2 / 4)*N_arr_Z

            # Length of one tie
            l_tr_y = Depth - 2 * parameters.cover - D_tr
            l_tr_z = Width - 2 * parameters.cover - D_tr

            Volume_tr = (A_sw_y*l_tr_y+A_sw_z*l_tr_z)*N_tr

            # Checking shear reinforcement requirement
            if A_sw_y/(spacing_tr*Width)<0.08*fck**(1/2)/parameters.fyk or A_sw_z/(spacing_tr*Depth)<0.08*fck**(1/2)/parameters.fyk:
                continue

            z_param_y =0.9*(Depth-parameters.cover-D_tr-D_rebar/2)  # Shear parameter (taken from the lecture, correct?)
            z_param_z =0.9*(Width-parameters.cover-D_tr-D_rebar/2)  # Shear parameter
            
            list_Vy_for_cot = V_ref*np.random.random(size=5)
            
            # Computing the shear capacity
            for each_Vy in list_Vy_for_cot:
                #V_y*(cotTheta_y^2)-cotTheta_y*v1*fcd*Width*z+V_y=0
                discr_y = ((v1*fcd*Width*z_param_y)**2)-4*each_Vy*each_Vy
                if discr_y<0:
                    continue
                cotTheta_y1=(v1*fcd*Width*z_param_y-(discr_y)**0.5)/(2*each_Vy)
                cotTheta_y2=(v1*fcd*Width*z_param_y+(discr_y)**0.5)/(2*each_Vy)
                
                if cotTheta_y1 >= 1.0 and cotTheta_y1<=2.5:
                    cotTheta_y = cotTheta_y1
                elif cotTheta_y2 >= 1.0 and cotTheta_y2<=2.5:
                    cotTheta_y = cotTheta_y2
                elif cotTheta_y1>=2.5 or cotTheta_y2>=2.5:
                    cotTheta_y = 2.5
                elif cotTheta_y1<=1.0 or cotTheta_y2<=1.0:
                    cotTheta_y = 1.0
                
                
                # Max shear capacity
                V_max_y = alpha_param * v1 * (-fcd) * Width * z_param_y / (cotTheta_y + 1/cotTheta_y)
         
                Vy = min(z_param_y * A_sw_y * fyd_tr * cotTheta_y / spacing_tr, V_max_y)
                if each_Vy>Vy:
                    continue
                                
                list_Vz_for_cot = V_ref*np.random.random(size=5)
                
                
                for each_Vz in list_Vz_for_cot:
                   
                    discr_z = ((v1*fcd*Depth*z_param_z)**2)-4*each_Vz*each_Vz
                    if discr_z<0:
                        continue
                    cotTheta_z1=(v1*fcd*Depth*z_param_z-(discr_z)**0.5)/(2*each_Vz)
                    cotTheta_z2=(v1*fcd*Depth*z_param_z+(discr_z)**0.5)/(2*each_Vz)
                    
                    if cotTheta_z1 >= 1.0 and cotTheta_z1<=2.5:
                        cotTheta_z = cotTheta_z1
                    elif cotTheta_z2 >= 1.0 and cotTheta_z2<=2.5:
                        cotTheta_z = cotTheta_z2
                    elif cotTheta_z1>=2.5 or cotTheta_z2>=2.5:
                        cotTheta_z = 2.5
                    elif cotTheta_z1<=1.0 or cotTheta_z2<=1.0:
                        cotTheta_z = 1.0

                    # Max shear capacity
                    V_max_z = alpha_param * v1 * (-fcd) * Depth * z_param_z / (cotTheta_z + 1/cotTheta_z)
             
                    Vz = min(z_param_z * A_sw_z * fyd_tr * cotTheta_z / spacing_tr, V_max_z)
                    if each_Vz>Vz:
                        continue
                
                    # *********************** END: Shear capacity **********************************
        
                    # Calculating the design axial load
                    # Decreasing the max axial load due to column concrete and steel weight
                    Steel_weight = Height*parameters.steel_weight*As                        # Longitudinal rebars weight
                    Tr_steel_weight = Volume_tr*parameters.steel_weight_tr                  # Transverse rebars weight
                    Concrete_weight = parameters.concrete_weight*(Height*Ac-Volume_tr)      # Concrete weight
                    Column_weight = Steel_weight + Tr_steel_weight + Concrete_weight        # Total column weight
                    Pmax = -parameters.alpha_coef*(parameters.nu_coef*(-fcd)*Ac+fyd*As)+Column_weight
        
                    # Checking reinforcement requirements
                    if As<0.002*A_core:
                            continue
        
                    # ***** START: Inner loop for increasing axial load P until failure *****
                    # Generating list of axial loads
                    list_P = Pmax * np.random.random(size=5)
        
                    # Calculating My capacity (uniaxial moment case)
                    list_My_max = []    # List to store the My capacity for each axial load P from list_P
                    for v in range(len(list_P)):
                        P = list_P[v]
                        # Checking reinforcement requirements
                        if -0.1*P/fyd>As:
                            continue
        
                        # Creating files to store the stress, strain and moment from four corner points of column section
                        strain1 = './out/strain11_' +str(i)+'_'+str(0)+'_'+ str(v)+'.txt'
                        strain2 = './out/strain12_' +str(i)+'_'+str(0)+'_'+ str(v)+'.txt'
                        strain3 = './out/strain13_' +str(i)+'_'+str(0)+'_'+ str(v)+'.txt'
                        strain4 = './out/strain14_' +str(i)+'_'+str(0)+'_'+ str(v)+'.txt'
                        strains = [strain1, strain2, strain3, strain4]
        
                        # Calling the section analysis procedure
                        MomentCurvature(parameters, P, Kz, -1, 5, strains, dist1, dist2)
        
                        # Creating a list to store the step when the ultimate strength strain is reached
                        indices = []
        
                        # Extracting step when the ultimate strain (eps2U) is reached
                        for k in range(4):
                            strain = strains[k]
                            if os.path.getsize(strain) > 0:
                                strain = pd.read_csv(strain, sep=' ', header=None)
                                filtered = strain[strain[2] >= eps2U]
                                if len(filtered) > 1:
                                    indices.append(list(filtered.index)[-1])
        
                        #delete_files(1, i, 0, v)
                        delete_files()
                        
                        # Saving the moment capacity (My) if the analysis converged
                        if len(indices) >= 1:
                            M_ult_index = min(indices)
                            M_ult = strain.loc[M_ult_index, [0]]
                            list_My_max.append(float(M_ult))
        
                            # Creating a list of moments in Y direction up to My capacity
                            list_m = float(M_ult)*np.random.random(size=5)
        
                            # Calculating Mz capacity (biaxial bending case)
                            for m in range(len(list_m)):
                                # Creating files to store the stress, strain and moment from 4 corner points of section
                                strain21 = './out/strain21_' + str(i)+'_'+str(v)+'_'+ str(m)+'.txt'
                                strain22 = './out/strain22_' + str(i)+'_'+str(v)+'_'+ str(m)+'.txt'
                                strain23 = './out/strain23_' + str(i)+'_'+str(v)+'_'+ str(m)+'.txt'
                                strain24 = './out/strain24_' + str(i)+'_'+str(v)+'_'+ str(m)+'.txt'
                                strains2 = [strain21, strain22, strain23, strain24]
        
                                # Calling the section analysis procedure to computer Mz capacity
                                MomentCurvature(parameters, P, Ky, list_m[m], 6, strains2, dist1, dist2)
        
                                # Creating a list to store the step when the ultimate strength strain is reached
                                indices = []
                                # Extracting step when the ultimate strain (eps2U) is reached
                                for k in range(4):
                                    strain = strains2[k]
                                    if os.path.getsize(strain) > 0:
                                        strain = pd.read_csv(strain, sep=' ', header=None, )
                                        filtered = strain[strain[2] >= eps2U]
                                        if len(filtered) > 1:
                                            indices.append(list(filtered.index)[-1])
        
                                #delete_files(2, i, v, m)
                                delete_files()
        
                                # Saving the moment capacity (Mz) if the analysis converged
                                if len(indices) >= 1:
                                
                                    M_ult_index = min(indices)
                                    M_ult = strain[0].values[M_ult_index]
                                    list_My_max.append(float(M_ult))

                                        
                                    # Filling the dictionary with the design parameters
                                    data['P'].append(P), data['My'].append(list_m[m]), data['Mz'].append(M_ult)
                                    data['fck'].append(-fck), data['h'].append(Height), data['Width'].append(Width)
                                    data['Depth'].append(Depth), data['D_rebar'].append(D_rebar), data['w_g'].append(w_g)
                                    data['d_g'].append(d_g), data['numRebars'].append(numTotRebars), data['As_total'].append(As)
    
                                    data['Vy'].append(Vy), data['Vz'].append(Vz)
                                    data['D_tr'].append(D_tr), data['s_tr'].append(spacing_tr)
                                    data['A_sw_y'].append(A_sw_y), data['A_sw_z'].append(A_sw_z)
                                    data['Volume_tr'].append(Volume_tr), data['N_tr'].append(N_tr)

        # Saving the design
        if i % numSaveToFile == 0:
            # Creating dataframe with the data from the dictionary of design parameters
            df = pd.DataFrame(data)

            # Dropping failure points
            df = df.dropna()
            df = df[(df['Mz'] > 0.0) & (df['My'] > 0.0)]

            # Saving the dataframe with designs to a csv file
            df.to_csv(fileName, mode='a', index=False, header=False)
            print("#%s: %s column designs already saved." % (z, i))

            # Cleaning the dictionary
            # data = empty_data_for_params()
            data = {'P': [], 'My': [], 'Mz': [], 'Vy': [], 'Vz': [], 'fck': [], 'h': [], 'Width': [], 'Depth': [],
                    'D_rebar': [],
                    'w_g': [], 'd_g': [], 'numRebars': [], 'As_total': [], 'D_tr': [], 's_tr': [], 'A_sw_y': [],
                    'A_sw_z': [],
                    'Volume_tr': [], 'N_tr': []}

        # Increasing counter by one for the next design
        i += 1
