import csv
import os
from functions import *
import column
import math
import numpy as np
import time

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Create an instance of class Column 
parameters = column.Column()

# Loading the dataset
filename = 'nn_out.csv' 
df = pd.read_csv(filename)

# Adding columns to store gridlines
df['w_g']=0
df['d_g']=0

# Directory to store the output response files
directory ='.'

# Lists to track the checks
numBar = []
As_final = []
As_1 = []
As_2 = []
Spacing = []
spacing_check = []
Shear_req = []
Vy_check = []
Vz_check = []
P_check = []
Mz_check = []
My_check = []

j = 0
for index, row in df.iterrows():
    print(index)

    # Loading the design predictions one by one
    P, My_red, Mz_red, Vy, Vz,fck, Height, Width, Depth, As_total, Volume_transverse, w_g, d_g= row

    # Change the direction of the axial load
    P = -P

    # Rounding the predicted width and depth as a multiple of 5 cm
    Width_top = 0.05*math.ceil(Width/0.05)
    Width_bottom = 0.05*math.floor(Width/0.05)

    Depth_top = 0.05*math.ceil(Depth/0.05)
    Depth_bottom = 0.05*math.floor(Depth/0.05)
    list_geom = [[Width_bottom,Width_bottom],[Width_bottom,Depth_top],[Width_top,Depth_bottom],[Width_top,Depth_top]]

    for Width, Depth in (list_geom):

        df.at[index, 'Width'] = Width
        df.at[index, 'Depth'] = Depth

        # Concrete parameters
        fcd = -fck/parameters.g_c     # Design compressive strength
        eps1U = parameters.eps1U
        eps2U = parameters.eps2U

        # Steel parameters
        fyd = parameters.fyk/parameters.g_s             # Design yield strength of longitudinal rebars
        fyd_tr = parameters.fyk_tr/parameters.g_s     # Design yield strength of transverse rebars

        # Selecting rebar diameter

        passed = 0

        # Checking for all reinforcement diameters until the first passing one is found
        for D_rebar in parameters.d_rebars:

            df.at[index, 'D_rebar'] = D_rebar
            # Calculating steel area and reinforcement constraints
            A_core = Width * Depth                      # Section gross area
            As_min = 0.002 * A_core                     # Min reinforcement area
            As_max = 0.04 * A_core                      # Max reinforcement area

            # Calculating area of one rebar
            As_bar = math.pi*(D_rebar**2)/4

            if As_total>=As_min and As_total<As_max:
                As_1.append(index)

                numBarsSec1 = math.floor(As_total/As_bar)
                numBarsSec2 = math.ceil(As_total/As_bar)
                numbarslist = [numBarsSec1, numBarsSec2]

                for numBarsSec in numbarslist:
                    if numBarsSec>=4:

                        numBar.append(index)

                        # Selecting the transverse rebar diameter
                        D_min = max(D_rebar/4, 0.006)                 # Eurocode page 162 section 9.5.3
                        a = [x - D_min for x in parameters.d_rebars]
                        D_index_min = a.index(min([i for i in a if i > 0]))

                        # Max tie diameter is assumed to be D_rebar
                        D_max = D_rebar
                        D_index_max = parameters.d_rebars_tr.tolist().index(D_max)


                        for diff_d_tr in range(D_index_max-D_index_min):
                            D_tr = parameters.d_rebars_tr[D_index_min+diff_d_tr]


                            # Geometry modelling parameters
                            coverY = Depth/2.0                          # Distance from the section z-axis to the edge of the cover concrete -- outer edge of cover concrete
                            coverZ = Width/2.0                          # Distance from the section y-axis to the edge of the cover concrete -- outer edge of cover concrete
                            coreY = coverY-parameters.cover-D_rebar/2-D_tr   # Distance from the section z-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
                            coreZ = coverZ-parameters.cover-D_rebar/2-D_tr   # Distance from the section y-axis to the edge of the core concrete --  edge of the core concrete/inner edge of cover concrete
                            dist1 = coverY-parameters.cover/2
                            dist2 = coverZ-parameters.cover/2

                            # Scheme with parametrisation using no reinforcement area
                            listDivisorPairs = returnDivisorsPair(numBarsSec)
                            if (len(listDivisorPairs) == 1):
                                listDivisorPairs = returnDivisorsPair(numBarsSec-1)

                            list_w_g = [*range(2, math.ceil(numBarsSec/2)+1, 1)]

                            for w_g in list_w_g:

                                d_g = math.ceil(numBarsSec/2)-w_g+2


                                w_h = (Width-2*D_rebar-2.5*parameters.cover-2*D_tr)/Width 
                                d_h = (Depth-2*D_rebar-2.5*parameters.cover-2*D_tr)/Depth

                                rebarZ = np.linspace(-coreZ, coreZ, w_g)
                                rebarY = np.linspace(-coreY, coreY, d_g)
                                spacingZ = (2*coreZ)/(w_g-1)
                                spacingY = (2*coreY)/(d_g-1)
                                # Checking for minimal spacing requirement
                                spacing_min = max(2*D_rebar, D_rebar+0.032+0.005, D_rebar+0.020)  # [m]
                                if spacingZ>=spacing_min and spacingY>=spacing_min:
                                    
                                    Spacing.append(index)
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
                                            if abs(Y) < hollowY and abs(Z) < hollowZ:
                                                continue
                                            rebars_YZ = np.vstack([rebars_YZ, [Y, Z]])
                                    for ii in range(len(rebars_YZ)):
                                        ops.fiber(*rebars_YZ[ii], As_bar, parameters.IDreinf)


                                    # Checking for number of rebars in final configuration
                                    numTotRebars = len(rebars_YZ)
                                    if numTotRebars>=4 or numTotRebars*As_bar>As_min:
                                        As_2.append(index)
                                        eps = fyd / parameters.Es   # Steel yield strain
                                        d_y = Width-parameters.cover-D_rebar/2-D_tr   # Distance from column outer edge to rebar in Y-axis
                                        d_z = Depth-parameters.cover-D_rebar/2-D_tr   # Distance from column outer edge to rebar in Z-axis
                                        Ky = eps/(0.7*d_y)      # Yield curvature in Y-axis
                                        Kz = eps/(0.7*d_z)      # Yield curvature in Z-axis

                                        As = As_bar*numTotRebars    # Total reinforcement area
                                        Ac = Width*Depth-As         # Total concrete area
                                        # *********************** START: Shear capacity **********************************

                                        # The number depends on the links arrangement
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
                                        s_min = 0.05 
                                        s_max = min(20 * D_rebar, Depth, Width, 0.4)    # Max spacing (see Eurocode 2 page 163)

                                        if s_min<s_max:
                                            spacing_check.append(index)

                                            # Cross-sectional area of the shear reinforcement
                                            A_sw_y = (math.pi * D_tr ** 2 / 4) * N_arr_Y
                                            A_sw_z = (math.pi * D_tr ** 2 / 4) * N_arr_Z
                                            # Length of one tie
                                            l_tr_y = Depth - 2 * parameters.cover - D_tr
                                            l_tr_z = Width - 2 * parameters.cover - D_tr

                                            # Computing the spacing for the selected design configuration
                                            spacing_tr_cl = 0.05*math.ceil((Height*(A_sw_y*l_tr_y+A_sw_z*l_tr_z)/Volume_transverse)/0.05)  # Number of ties in a column
                                            spacing_tr_fl = 0.05*math.floor((Height*(A_sw_y*l_tr_y+A_sw_z*l_tr_z)/Volume_transverse)/0.05)
                                            if spacing_tr_fl==0:
                                                spacing_tr_fl = 0.05
                                            if spacing_tr_cl == 0:
                                                spacing_tr_cl = 0.05
                                            spacing_tr_list = [spacing_tr_cl, spacing_tr_fl]
                                            for spacing_tr in spacing_tr_list:


                                                N_tr_fl = math.floor(Height / spacing_tr)  # Number of ties in a column
                                                N_tr_cl = math.ceil(Height / spacing_tr)
                                                N_tr_list = [N_tr_fl, N_tr_cl]

                                                for N_tr in N_tr_list:
                                                    Volume_tr = (A_sw_y*l_tr_y+A_sw_z*l_tr_z)*N_tr


                                                    # Checking shear reinforcement requirement
                                                    if A_sw_y/(spacing_tr*Width)>=0.08*fck**(1/2)/parameters.fyk and A_sw_z/(spacing_tr*Depth)>=0.08*fck**(1/2)/parameters.fyk:

                                                        Shear_req.append(index)
                                                        z_param_y =0.9*(Depth-parameters.cover-D_tr-D_rebar/2)  # Shear parameter (taken from the lecture, correct?)
                                                        z_param_z =0.9*(Width-parameters.cover-D_tr-D_rebar/2)  # Shear parameter

                                                        
                                                        
                                                        discr_y = ((v1*fcd*Width*z_param_y)**2)-4*Vy*Vy
                                                        if discr_y<0:
                                                            continue
                                                        cotTheta_y1=(v1*fcd*Width*z_param_y-(discr_y)**0.5)/(2*Vy)
                                                        cotTheta_y2=(v1*fcd*Width*z_param_y+(discr_y)**0.5)/(2*Vy)

                                                        if cotTheta_y1 >= 1.0 and cotTheta_y1<=2.5:
                                                            cotTheta_y = cotTheta_y1
                                                        elif cotTheta_y2 >= 1.0 and cotTheta_y2<=2.5:
                                                            cotTheta_y = cotTheta_y2
                                                        elif cotTheta_y1>=2.5 or cotTheta_y2>=2.5:
                                                            cotTheta_y = 2.5
                                                        elif cotTheta_y1<=1.0 or cotTheta_y2<=1.0:
                                                            cotTheta_y = 1.0

                                                        discr_z = ((v1*fcd*Depth*z_param_z)**2)-4*Vz*Vz
                                                        if discr_z<0:
                                                            continue
                                                        cotTheta_z1=(v1*fcd*Depth*z_param_z-(discr_z)**0.5)/(2*Vz)
                                                        cotTheta_z2=(v1*fcd*Depth*z_param_z+(discr_z)**0.5)/(2*Vz)

                                                        if cotTheta_z1 >= 1.0 and cotTheta_z1<=2.5:
                                                            cotTheta_z = cotTheta_z1
                                                        elif cotTheta_z2 >= 1.0 and cotTheta_z2<=2.5:
                                                            cotTheta_z = cotTheta_z2
                                                        elif cotTheta_z1>=2.5 or cotTheta_z2>=2.5:
                                                            cotTheta_z = 2.5
                                                        elif cotTheta_z1<=1.0 or cotTheta_z2<=1.0:
                                                            cotTheta_z = 1.0

                                                        # Max shear capacity
                                                        V_max_y = alpha_param * v1 * (-fcd) * Width * z_param_y / (cotTheta_y + 1/cotTheta_y)
                                                        V_max_z = alpha_param * v1 * (-fcd) * Depth * z_param_z / (cotTheta_z + 1/cotTheta_z)

                                                        Vy_temp = min(z_param_y * A_sw_y * fyd_tr * cotTheta_y / spacing_tr, V_max_y)
                                                        Vz_temp = min(z_param_z * A_sw_z * fyd_tr * cotTheta_z / spacing_tr, V_max_z)

                                                        if Vy<Vy_temp:
                                                            Vy_check.append(index)
                                                            if Vz<Vz_temp:
                                                                Vz_check.append(index)

                                                                Steel_weight = Height*parameters.steel_weight*As                        # Longitudinal rebars weight
                                                                Tr_steel_weight = Volume_tr*parameters.steel_weight_tr                  # Transverse rebars weight
                                                                Concrete_weight = parameters.concrete_weight*(Height*Ac-Volume_tr)      # Concrete weight
                                                                Column_weight = Steel_weight + Tr_steel_weight + Concrete_weight        # Total column weight
                                                                Pmax = -parameters.alpha_coef*(parameters.nu_coef*(-fcd)*Ac+fyd*As)+Column_weight

                                                                # Checking reinforcement requirements
                                                               
                                                                if -0.1*P/fyd<As and As>0.002*A_core:
                                                                    As_final.append(index)
                                                                    
                                                                    # Check for axial load capacity
                                                                    if Pmax<P:
                                                                        # Record the axial load check pass
                                                                        P_check.append(index)

                                                                        j+=1
                                                                        # Creating files to store the stress, strain and moment from four corner points of column section
                                                                        strain1 = './output/strain11'+str(index)+str(j)+'.txt'
                                                                        strain2 = './output/strain12'+str(index)+str(j)+'.txt'
                                                                        strain3 = './output/strain13'+str(index)+str(j)+'.txt'
                                                                        strain4 = './output/strain14'+str(index)+str(j)+'.txt'
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


                                                                        # Saving the moment capacity (My) if the analysis converged
                                                                        if len(indices) >= 1:
                                                                            My_ult_index = min(indices)
                                                                            My_ult = strain.loc[My_ult_index, [0]]
                                                                            
                                                                            if My_red<My_ult[0]:
                                                                                My_check.append(index)
                                                                                j+=1
                                                                                # Creating files to store the stress, strain and moment from 4 corner points of section
                                                                                strain21 = './output/strain21'+str(index)+str(j)+'.txt'
                                                                                strain22 = './output/strain22'+str(index)+str(j)+'.txt'
                                                                                strain23 = './output/strain23'+str(index)+str(j)+'.txt'
                                                                                strain24 = './output/strain24'+str(index)+str(j)+'.txt'
                                                                                strains2 = [strain21, strain22, strain23, strain24]


                                                                                # Calling the section analysis procedure to computer Mz capacity
                                                                                MomentCurvature(parameters, P, Ky, My_red, 6, strains2, dist1, dist2)

                                                                                # Creating a list to store the step when the ultimate strength strain is reached
                                                                                indices1 = []
                                                                                # Extracting step when the ultimate strain (eps2U) is reached
                                                                                for k in range(4):
                                                                                    strain = strains2[k]
                                                                                    if os.path.getsize(strain)>0:
                                                                                        strain = pd.read_csv(strain, sep=' ', header=None, )
                                                                                        filtered = strain[strain[2]>=eps2U]
                                                                                        if len(filtered)>1:
                                                                                            indices1.append(list(filtered.index)[-1])


                                                                                # Saving the moment capacity (Mz) if the analysis converged
                                                                                if len(indices1)>=1:
                                                                                    M_ult_index = min(indices1)
                                                                                    Mz_ult = strain[0].values[M_ult_index]

                                                                                    if Mz_red<=float(Mz_ult):
                                                                                        Mz_check.append(index)
                                                                                        passed = 1
                                                                                        df.at[index, 'w_g'] = w_g
                                                                                        df.at[index, 'd_g'] = d_g
                                                                                        df.at[index, 'D_tr'] = D_tr
                                                                                        df.at[index, 'Width'] = Width
                                                                                        df.at[index, 'Depth'] = Depth
                                                                                        df.at[index, 'As_total'] = As
                                                                                        df.at[index, 'Volume_tr'] = Volume_tr
                                                                                        index += 1
                                                                                        
                                                                                        print("#############")
                                                                                        break

                                                    if passed==1:
                                                        break
                                                if passed==1:
                                                    break
                                if passed==1:
                                    break                                                                          
                       
                            if passed==1:
                                break    
                        if passed==1:
                            break
            if passed==1:
                break

        if passed==1:
            break
print(len(set(numBar)), len(set(Spacing)), len(set(As_2)), len(set(spacing_check)), len(set(Shear_req)))
print(len(set(Vy_check)), len(set(Vz_check)), len(set(P_check)), len(set(As_final)), len(set(My_check)), len(set(Mz_check)))               
                                                            
                    
# Compute the cost of the final designs
df['fck'] = (-1)*df['fck']
fcs = np.asarray([-50.0, -45.0, -40.0, -35.0, -30.0, -25.0])

def price_concrete(row):
    # Source for prices: https://jcbetons.lv/cenas-en/?lang=en
    # + 21% VAT
    if row['fck'] == fcs[0]:
        return 95 * 1.21  # 95 EUR/m3 - assumed value
    if row['fck'] == fcs[1]:
        return 90 * 1.21  # 90 EUR/m3 - assumed value
    if row['fck'] == fcs[2]:
        return 85 * 1.21  # 85 EUR/m3 - assumed value
    if row['fck'] == fcs[3]:
        return 72 * 1.21
    if row['fck'] == fcs[4]:
        return 63 * 1.21
    if row['fck'] == fcs[5]:
        return 60 * 1.21
    return -1

# Prices in EUR/m3
df['price_s'] = 0.31* 1.21*7850
df['price_c'] = df.apply(lambda row: price_concrete(row), axis=1)

df['price'] = df['h']*((df['Width']*df['Depth'] - df['As_total'])*df['price_c'] + df['As_total']*df['price_s'])+df['Volume_tr']*df['price_s']

df.to_csv(file+"_price.csv")


