import openseespy.opensees as ops
import numpy as np
import math
import random
import pandas as pd
import os
import os.path
from os import path
from pathlib import Path
import time
import subprocess



def empty_data_for_params():
    data = {'P': [], 'My': [], 'Mz': [], 'Vy': [], 'Vz': [], 'fck': [], 'h': [], 'Width': [], 'Depth': [],
            'D_rebar': [],
            'w_g': [], 'd_g': [], 'numRebars': [], 'As_total': [], 'D_tr': [], 's_tr': [], 'A_sw_y': [], 'A_sw_z': [],
            'Volume_tr': [], 'N_tr': []}
    return data

def cross_section(Height):
    # =============================================================================
    #     According to 5.3.1.7 Eurocode 2, 
    #     A column is a member for which the section depth does not exceed 4 times its width and
    #     the height is at least 3 times the section depth. Otherwise it should be considered as a wall. (page 57)
    # =============================================================================
    widthArray = [*range(round(0.2/0.05), round((Height/3+0.05)/0.05), 1)]
    widthArray  = [round(0.05*x, 2) for x in widthArray]
    colWidth = random.choice(widthArray)
    # spacing_tr = random.randrange(1000 * s_min, 1000 * s_max, 5) / 1000
   
    depthArray = [*range(round(max(0.2, colWidth/4)/0.05), round((min(colWidth*4, Height/3)+0.05)/0.05), 1)]
    depthArray = [round(0.05*x, 2) for x in depthArray]
    colDepth = random.choice(depthArray)

    Width = min(colWidth, colDepth)
    Depth = max(colWidth, colDepth)
    return Width, Depth


# Degree of fredom (dof) = 5 for uniaxial moment
# Degree of fredom (dof) = 6 for biaxial moment
def MomentCurvature(parameters, axialLoad, maxK, m, dof, strains, dist1, dist2):
    """
    This is the function for column section analysis. For uniaxial case degree of freedom dof=5,
    for biaxial case dof=6. Inputs are object of class parameters, axial load P, maximum curvature,
    first direction moment, degree of freedom, list of txt files, the absolute value of edge point coordinates.
    Function stores the stress, strain and moment response to txt files.
    """
    # Define two nodes at (0,0) for zero-length element
    ops.node(1001, 0.0, 0.0, 0.0)
    ops.node(1002, 0.0, 0.0, 0.0)

    # Fix all degrees of freedom except axial and bending
    ops.fix(1001, 1, 1, 1, 1, 1, 1)
    ops.fix(1002, 0, 1, 1, 1, 0, 0)
    
    # Define element
    ops.element('zeroLengthSection', 2001, 1001, 1002, parameters.SecTag)

    # Create recorder to extract the response
    ops.recorder('Element', '-file', strains[0], '-time', '-ele', 2001, 'section', 'fiber', dist1, -dist2, 'stressStrain')
    ops.recorder('Element', '-file', strains[1], '-time', '-ele', 2001, 'section', 'fiber', -dist1, dist2, 'stressStrain')
    ops.recorder('Element', '-file', strains[2], '-time', '-ele', 2001, 'section', 'fiber', dist1, dist2, 'stressStrain')
    ops.recorder('Element', '-file', strains[3], '-time', '-ele', 2001, 'section', 'fiber', -dist1, -dist2, 'stressStrain')

    # Define constant axial load
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 3001, 1)
    ops.load(1002, axialLoad, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Defining the first direction moment
    if dof == 6:
        ops.timeSeries('Constant', 3)
        ops.pattern('Plain', 3003, 3)
        ops.load(1002, 0.0, 0.0, 0.0, 0.0, -m, 0.0)

    # Defining analysis parameters
    ops.integrator('LoadControl', 0, 1, 0, 0)
    ops.system('SparseGeneral', '-piv')
    ops.test('EnergyIncr', 1e-9, 10)
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.algorithm('Newton')
    ops.analysis('Static')

    # Analyzing for constant axial load
    ops.analyze(1)
    ops.loadConst('-time', 0.0)

    # Defining reference moment
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 3002, 2)

    if dof == 6:
        ops.load(1002, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) 
    else:
        ops.load(1002, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0) 

    # Computing curvature increment
    dK = maxK*parameters.mu/parameters.numIncr

    # Using displacement control at node 2 for section analysis
    ops.integrator('DisplacementControl', 1002, dof, dK, 1, dK, dK)

    # Analyzing the section
    ops.analyze(parameters.numIncr)

    # CLeaning cash after the analysis
    ops.wipeAnalysis()
    ops.remove('node', 1001), ops.remove('node', 1002), ops.remove('ele', 2001), ops.remove('sp', 1001, 1)
    ops.remove('sp', 1001, 2), ops.remove('sp', 1001, 3), ops.remove('sp', 1001, 4), ops.remove('sp', 1001, 5)
    ops.remove('sp', 1001, 6), ops.remove('sp', 1002, 1), ops.remove('sp', 1002, 2), ops.remove('sp', 1002, 3)
    ops.remove('sp', 1002, 4), ops.remove('sp', 1002, 5), ops.remove('sp', 1002, 6), ops.remove('timeSeries', 1)
    ops.remove('loadPattern', 3001), ops.remove('timeSeries', 2), ops.remove('loadPattern', 3002)
    
    if dof == 6:
        ops.remove('timeSeries', 3)
        ops.remove('loadPattern', 3003)

    # Cleaning recorders
    ops.remove('recorders')
    

def returnDivisorsPair(n):
    """
    This function finds all divisor pairs for rebars. Input is selected number of rebars.
    Output is list of sublist with divisor pairs.
    """
    listDivisors = []
    # Note: this loop runs till square root
    i = 1
    while i <= math.sqrt(n):
        if n%i == 0:
            # If divisors are equal, print only one
            # Otherwise print both
            if n/i == i:
                listDivisors.append((i, i))
            else:
                listDivisors.append((i, n/i))
        i = i + 1
    return listDivisors


def grid_params(listDivisorPairs):
    """
    This function randomly select a list of pairs to construct the grid for rebar placement.
    The input is a list of sublists, where each list contains divisors of the selected number of rebars.
    Outputs are grid parameters: w_g and d_g
    """
    gridDimsPair = listDivisorPairs[random.randint(1, len(listDivisorPairs)-1)]
    randomGridDimAssign = random.sample(range(0, 2), 2)
    w_g = gridDimsPair[randomGridDimAssign[0]]
    d_g = gridDimsPair[randomGridDimAssign[1]]
    return w_g, d_g



def delete_files ():#(n, i, vv, mm):
	"""
    This function deletes the txt files created to store the stress-strain analysis 
    after the data has been extracted to empty the storage
    """
    now = time.time()
    
    folder = './out/'
    
    files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    for filename in files:
        if (now - os.stat(filename).st_mtime) > 120:
            command = "rm {0}".format(filename)
            subprocess.call(command, shell=True)
            os.remove(filename)
