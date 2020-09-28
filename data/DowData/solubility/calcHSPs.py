#!/usr/bin/python
# -*- coding: utf-8 -*-

# script file to compute HSP components of pure compounds
# note that the rdkit library is required
# input: MDL .mol files
# output: delta(d) delta(p) delta(h) Vm are printed out in this order
# units: MPa**1/2 and cc/mol
# Taken from : https://acs.figshare.com/articles/Pencil_and_Paper_Estimation_of_Hansen_Solubility_Parameters/7449257/1

import sys, os, fileinput
from math import sqrt
from collections import Counter, OrderedDict, defaultdict
from rdkit import Chem

# FOR MOLAR VOLUME AND REFRACTIVITY

# additive increments
params = (('B30', 7.61, 2.91),
          ('Br10', 27.52, 8.44),
          ('C10', 14.68, 2.32),
          ('C20', 9.89, 4.01),
          ('C21', 22.77, 4.58),
          ('C30', -0.00, 3.15),
          ('C30a', -0.00, 3.48),
          ('C31', 13.23, 4.65),
          ('C31a', 13.23, 4.46),
          ('C32', 27.17, 5.38),
          ('C40', -8.40, 2.60),
          ('C41', 4.46, 3.48),
          ('C42', 16.57, 4.60),
          ('C43', 29.58, 5.74),
          ('Cl10', 24.74, 5.87),
          ('F10', 17.63, 1.06),
          ('Ge40', 9.36, 8.54),
          ('I10', 35.64, 13.75),
          ('N10', 14.09, 1.55),
          ('N20', 7.42, 2.60),
          ('N20a', 7.42, 2.44),
          ('N21', 18.14, 3.55),
          ('N30', -3.08, 2.89),
          ('N30a', -3.08, 2.85),
          ('N31', 7.74, 3.69),
          ('N31a', 7.74, 3.72),
          ('N32', 17.81, 4.60),
          ('N43', 5.36, 8.02),
          ('O10', 14.89, 1.84),
          ('O20', 6.25, 1.55),
          ('O20a', 6.25, 0.71),
          ('O21', 11.78, 2.51),
          ('P30', 10.42, 7.41),
          ('P40', -1.94, 4.98),
          ('P41', 10.06, 5.41),
          ('R5', 9.41, None),
          ('R6', 6.89, None),
          ('R<5', 10.89, None),
          ('R>6', 3.75, None),
          ('S10', 25.92, 10.60),
          ('S20', 14.90, 8.22),
          ('S20a', 14.90, 7.05),
          ('S21', 26.14, 8.71),
          ('S30', 5.58, 7.28),
          ('S40', -3.74, 5.12),
          ('Se20', 19.00, 11.53),
          ('Se20a', 19.00, 9.43),
          ('Si40', 9.28, 6.38),
          ('Si41', 23.35, 7.85),
          ('Sn40', 14.47, 14.90),
          ('Ti40', 6.09, 14.66),
          ('aromat', 1.82, None))
vi = dict((p[0], p[1]) for p in params)  # volume increments
ri = dict((p[0], p[2]) for p in params)  # molar refractivity increments

def get_atom_code(atom):
    """ return an atom code consistent with the keys of the 'Vinc' dictionary """ 
    # be careful to take into account nD = number of deuterium atoms !
    nD = len([x for x in atom.GetNeighbors() if x.GetMass()>2 and x.GetSymbol()=='H'])
    if nD > 0:
        print('nD %u %s' %(nD, arg))
    code = atom.GetSymbol() + str(atom.GetTotalDegree()) + str(atom.GetTotalNumHs()+nD)
    code += 'a'*int(atom.GetIsAromatic())
    return code

def get_ring_descriptors(mol, maxi=6, mini=5):
    """ return dict of ring descriptors for molecule provided as input """
    dic = Counter()
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        size = len(ring)
        if size > maxi:
            label = 'R>'+str(maxi)
        elif size < mini:
            label = 'R<'+str(mini)
        else:
            label = 'R%u' %len(ring)
        dic[label] += 1
        # contribute also +1 aromatic ring ?
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if all(at.GetIsAromatic() for at in atoms):
            dic['aromat'] += 1
    return dic

def get_Vm_RD(mol):         
    atoms = Counter(get_atom_code(atom) for atom in mol.GetAtoms())
    missing_atoms = set(atoms) - set(vi)
    if missing_atoms: return None, None
    rings = get_ring_descriptors(mol)
    Vm = sum([atoms[k]*vi[k] for k in atoms])   # atoms contribution
    Vm += sum([rings[k]*vi[k] for k in rings])  # rings contribution
    RD = sum([atoms[k]*ri[k] for k in atoms])   # molar refraction
    return Vm, RD
    
# FOR POLAR (P) HSP COMPONENT

params_p = {
    'N(1)': 2783,
    'N(2)': 8235,
    'O(0)': 1603,
    'O(1)': 4125,
    'Cl(0)': 1637,
    'C=O': 7492,
    'COOH': -5494,
    'COinAmide': 15972,
    'Carbonate': 19019,
    'Ester': 3653,
    'C#N': 16056,
    'NitroOnC': 13276,
    'O=P': 20310,
}

POLARGROUPS = OrderedDict([
  ("COOH", ("[CX3](=O)[OX2H1]"                          , (1,2))),  # =O in carboxilic acid
  ("NitroOnC", ("[#6][$([NX3](=O)=O),$([NX3+](=O)[O-])]", (1,))),  # N in nitroaliphatic
  ("Carbonate", ("[OX2;R][CX3;R](=O)[OX2]"              , (0,1,3))),  # Carbonate
  ("Ester", ("[OX2][CX3]=O"                             , (1,))),  # C in cyclic ester
  ("COinAmide", ("[NX3][CX3](=[OX1])[#6]"               , (1,))),
  ("SO2", ("O=S=O"                                      , (1,))),  # S in sulfone
])
    
def GetBChar(bond):
    if bond.GetBondType()==Chem.rdchem.BondType.AROMATIC: return '~'
    if bond.GetBondType()==Chem.rdchem.BondType.DOUBLE: return '='
    if bond.GetBondType()==Chem.rdchem.BondType.TRIPLE: return '#'
    return '-'

def get_nH(atom):
    nD = len([x for x in atom.GetNeighbors() if x.GetMass()>2 and x.GetSymbol()=='H'])
    nH = nD + atom.GetTotalNumHs()
    return nH

def isNitroN(at, mol):
    if at.GetSymbol() != "N": return False
    if at.GetTotalDegree() != 3: return False
    voisins = at.GetNeighbors()
    Os = [a for a in voisins if a.GetSymbol()=="O"]
    O1s = [a for a in Os if a.GetTotalDegree()==1]
    return len(O1s) > 1
    
def isAmideN(at, mol):
    amideSMARTS = "[NX3][CX3](=[OX1])[#6]"
    amidePattern = Chem.MolFromSmarts(amideSMARTS)
    N_indices = [t[0] for t in mol.GetSubstructMatches(amidePattern)]
    return at.GetIdx() in N_indices
    
def inCOOH(at, mol):
    acSMARTS = "[CX3](=O)[OX2H1]"
    acPattern = Chem.MolFromSmarts(acSMARTS)
    OH_indices = [t[2] for t in mol.GetSubstructMatches(acPattern)]
    return at.GetIdx() in OH_indices

def get_polar_groups(mol):
    global POLARGROUPS
    by_type = defaultdict(list)
    counted_indices = set()
    # first count complex polar groups
    for group_name in POLARGROUPS:
        group, positions = POLARGROUPS[group_name]
        pattern = Chem.MolFromSmarts(group)
        tuples = mol.GetSubstructMatches(pattern)
        for tup in tuples:
            if set(tup) & counted_indices: continue
            counted_indices |= set(tup)
            by_type[group_name].append(1)
    # count insaturated polar bonds
    for bond in mol.GetBonds():
        order = GetBChar(bond)
        if order in ("#", "=", "~"):
            abeg, aend = bond.GetBeginAtom(), bond.GetEndAtom()
            tup = (abeg.GetIdx(), aend.GetIdx())
            if set(tup) & counted_indices: continue
            symbols = sorted([abeg.GetSymbol(), aend.GetSymbol()])
            if set(symbols) == set(["C"]): continue
            bondsymbol = order.join(symbols)
            counted_indices |= set(tup)
            by_type[bondsymbol].append(order)
    # count saturated heteroatoms
    for hetat in mol.GetAtoms():
        idx = hetat.GetIdx()
        if idx in counted_indices: continue
        coo =  hetat.GetTotalDegree()
        symbol = hetat.GetSymbol()
        if symbol == "C": continue
        if symbol == "P" and coo > 3: continue
        name = "%s(%u)" %(symbol, get_nH(hetat))
        if name in ("N(0)", "F(0)"): continue
        counted_indices.add(idx)
        by_type[name].append(idx)
    return dict((group_name, len(by_type[group_name])) for group_name in by_type)

def get_polar_groups(mol):
    global POLARGROUPS
    by_type = defaultdict(list)
    counted_indices = set()
    for group_name in POLARGROUPS:
        group, positions = POLARGROUPS[group_name]
        pattern = Chem.MolFromSmarts(group)
        tuples = mol.GetSubstructMatches(pattern)
        for tup in tuples:
            pos = positions[0]
            atomindex = tup[pos]
            ##>> if set(tup) & counted_indices: continue
            counted_indices |= set(tup)
            by_type[group_name].append(atomindex)
    for bond in mol.GetBonds():
        order = GetBChar(bond)
        if order in ("#", "=", "~"):
            abeg, aend = bond.GetBeginAtom(), bond.GetEndAtom()
            symbols = sorted([abeg.GetSymbol(), aend.GetSymbol()])
            if symbols[0] == symbols[1]: continue  # ADDED to remove C=C and C~C
            tup = (abeg.GetIdx(), aend.GetIdx())
            if set(tup) & counted_indices: continue
            bondsymbol = order.join(symbols)
            counted_indices |= set(tup)
            by_type[bondsymbol].append(order)
    for hetat in mol.GetAtoms():
        idx = hetat.GetIdx()
        ##>> if idx in counted_indices: continue
        coo =  hetat.GetTotalDegree()
        symbol = hetat.GetSymbol()
        #if symbol == "C":
            #voisins = hetat.GetNeighbors()
            #Fs = [v for v in voisins if v.GetSymbol()=="F"]
            #if len(Fs) == 3:
                #by_type["CF3"].append(idx)
        if symbol == "C": continue
        if symbol == "P" and coo > 3: continue
        name = "%s(%u)" %(symbol, get_nH(hetat))
        if name in ("N(0)", "F(0)"): continue
        counted_indices.add(idx)
        by_type[name].append(idx)
    return dict((group_name, len(by_type[group_name])) for group_name in by_type)
    
    
    
def get_HSPp(mol):
    d = get_polar_groups(mol)
    if set(d) - set(params_p): return None
    eP = sum(params_p[k]*d[k] for k in d)
    return eP
    
# FOR HYDROGEN-BONDING (H) HSP COMPONENT

params_h = {
    'HC': 24.5,
    'HN': -1576,
    'HNamide': 5060,
    'H2N': 5484,
    'HO': 16945,
    'HO_COOH': 7094,
    'N': 3252,
    'O': 1980,
    'X': 412,
}

def get_dic_h(mol):
    dic = defaultdict(int)
    for at in mol.GetAtoms():
        symbol = at.GetSymbol()
        coo = at.GetTotalDegree()
        voisins = [v for v in at.GetNeighbors()]
        if symbol in ("N", "O"):
            if not isNitroN(at, mol):
                dic[symbol] += 1
        if symbol in ("F", "Cl", "Br", "I"):
            dic["X"] += 1
        nH = get_nH(at)
        if nH == 0: continue
        if symbol == "C":
            dic["HC"] += nH
            continue
        if symbol == "N":
            if isAmideN(at, mol):
                dic["HNamide"] += nH
                continue
            if nH == 2:
                dic['H2N'] += 2
            elif nH == 1:
                dic["HN"] += 1
            continue
        if symbol == "O":
            if inCOOH(at, mol):
                dic["HO_COOH"] += nH
            else:
                dic["HO"] += nH
    return dic

def get_HSPh(mol):
    d = get_dic_h(mol)
    if set(d) - set(params_h): return None
    eH = sum(params_h[k]*d[k] for k in d)
    return eH
   
# UTILITY FOT OUTPUT
   
def tostr(energy, volume=None):
    if None in (energy, volume): return "  xxxx "
    if volume == 0: return " %5.1f " %energy
    return " %5.1f " %sqrt(max(energy/volume,0))
   
# MAIN PROGRAM STARTS HERE

if __name__ == "__main__":
    for entry in fileinput.input():
        molsrc = entry.rstrip()
        # read molecule
        if not os.path.isfile(molsrc):
            mol = Chem.MolFromSmiles(molsrc)
        else:
            mol = Chem.MolFromMolFile(molsrc)
        molname = molsrc.split("/")[-1].replace(".mol", "")
        if mol is None:
            print("%s => None" %molname)
            continue
        # compute molar volume and refractivity
        Vm, RD = get_Vm_RD(mol)
        # compute D-component of HSP
        if Vm is None or RD is None:
            hspD = None
        else:
            hspD = sqrt(93.8 + (2016 + 75044./Vm) * (RD/Vm)**2)
        # compute P-component of HSP
        eP = get_HSPp(mol)
        eH = get_HSPh(mol)
        # convert energies to HSP
        hspD, hspP, hspH =  tostr(hspD,0), tostr(eP,Vm), tostr(eH,Vm)
        if Vm is not None:
            line = molname.ljust(12) + hspD + hspP + hspH + " %7.1f" %Vm
        else:
            line = molname.ljust(12) + hspD + hspP + hspH + "   None"
        print(line)
