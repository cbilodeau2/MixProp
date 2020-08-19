import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import sys
import os
from rdkit import Chem
from rdkit.Chem import Lipinski
import thermo
from thermo.chemical import Chemical
from thermo import Joback
from thermo import dipole
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from calcHSPs import *

def lipinski_descriptors(mol):
    descriptors =[]
    descriptors.append(Lipinski.FractionCSP3(mol))
    descriptors.append(Lipinski.NHOHCount(mol))
    descriptors.append(Lipinski.NOCount(mol))
    descriptors.append(Lipinski.NumAliphaticCarbocycles(mol))
    descriptors.append(Lipinski.NumAliphaticHeterocycles(mol))
    descriptors.append(Lipinski.NumAliphaticRings(mol))
    descriptors.append(Lipinski.NumAromaticCarbocycles(mol))
    descriptors.append(Lipinski.NumAromaticHeterocycles(mol))
    descriptors.append(Lipinski.NumAromaticRings(mol))
    descriptors.append(Lipinski.NumHAcceptors(mol))
    descriptors.append(Lipinski.NumHDonors(mol))
    descriptors.append(Lipinski.NumRotatableBonds(mol))
    descriptors.append(Lipinski.NumSaturatedCarbocycles(mol))
    descriptors.append(Lipinski.NumSaturatedHeterocycles(mol))
    descriptors.append(Lipinski.NumSaturatedRings(mol))
    descriptors.append(Lipinski.RingCount(mol))
    return descriptors  

def thermo_descriptors(smiles):
    try:
        chem = Chemical(smiles)
        descriptors = thermo.solubility.solubility_parameter(T=298.2, Hvapm=chem.Hvapm, Vml=chem.Vml)
        return descriptors
    except:
        pass
    
def joback_descriptors(smiles):
    try:
        J = Joback(smiles)
        descriptors = [J.Tb(J.counts),J.Tm(J.counts),J.Tc(J.counts),J.Hfus(J.counts),J.Hvap(J.counts)]
        return descriptors
    except:
        pass
    
def thermo_dipole(smiles):
    try:
        cas = Chemical(smiles).CAS
        descriptors = dipole.dipole_moment(cas)
        return descriptors
    except:
        pass
    
def thermo_surface_tension(smiles):
    try:
        st = Chemical(smiles).SurfaceTension(T=298)
        return st
    except:
        pass

def thermo_viscosity(smiles):
    try:
        v = Chemical(smiles).ViscosityLiquid(T=298,P=101325)
        return v
    except:
        pass

def thermo_skin_toxicity(smiles):
    try:
        tox = Chemical(smiles).Skin
        tox = (int(tox))
        return (tox)
    except:
        pass

def thermo_density(smiles):
    try:
        chem = Chemical(smiles)
        features = [chem.rho, chem.rhoc, chem.rhocm, chem.rhog, chem.rhogm, chem.rhol, chem.rholm, chem.rhom]
        if not None in features:
            return features
        else:
            pass
    except:
        pass

def thermo_flash(smiles):
    try:
        v = Chemical(smiles).Tflash
        return v
    except:
        pass

def thermo_hvap(smiles):
    try:
        chem = Chemical(smiles)
        features = [chem.Hvap, chem.Hvap_Tb, chem.Hvap_Tbm, chem.Hvapm]
        if not None in features:
            return features
        else:
            pass
    except:
        pass
    
def thermo_melting(smiles):
    try:
        t = Chemical(smiles).Tm
        return t
    except:
        pass

def thermo_boiling(smiles):
    try:
        t = Chemical(smiles).Tb
        return t   
    except:
        pass

def all_thermo(smiles):
    try:
        chem = Chemical(smiles)
        descriptors = [thermo.solubility.solubility_parameter(T=298.2, Hvapm=chem.Hvapm, Vml=chem.Vml)]
        descriptors = descriptors + [chem.SurfaceTension(T=298), chem.ViscosityLiquid(T=298,P=101325), int(chem.Skin),chem.Tflash, chem.Tm, chem.Tb]
        descriptors = descriptors + [chem.rho, chem.rhoc, chem.rhocm, chem.rhog, chem.rhogm, chem.rhol, chem.rholm, chem.rhom]
        descriptors = descriptors + [chem.Hvap, chem.Hvap_Tb, chem.Hvap_Tbm, chem.Hvapm]
        return descriptors
    except:
        pass

def morgan_descriptors(mol):
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        array = np.zeros((0, ), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array        
    except:
        pass
    
def rdkit_fingerprint(mol):
    try:
        fp =  Chem.RDKFingerprint(mol)
        array = np.zeros((0, ), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array
    except:
        pass    
    
    
def hansen_parameters(mol):
    try:
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

        hansen_data=[hspD,hspP,hspH,Vm]
        if not '  xxxx ' in hansen_data:
            return hansen_data
        else:
            pass
    except:
        pass

def generate_starting_files(infile='solubility_G3_G5.csv',
                           out_folder='Hansen_',
                           descriptor_type='hansen'): ## lipinski, joback, thermo, thermo_dipole, morgan,rdkit, hansen,thermo_viscosity,thermo_surface_tension,thermo_skin_toxicity
                            ## thermo_density, thermo_flash, thermo_hvap, thermo_melting, thermo_boiling, all_thermo
    # Toggle settings:
    remove_multiple_molecules = True # Typically true
    
    
    try:
        os.mkdir(out_folder)
    except:
        pass
    
    data = pd.read_csv(infile)
    
    if remove_multiple_molecules:
        data = data[[ not bool('.' in x) for x in data['SMILES']]]
    
    data['mol'] = data['SMILES'].apply(Chem.MolFromSmiles)
    
    # Lipinski Descriptors:
    if descriptor_type == 'lipinski':
        data['descriptors'] = data['mol'].apply(lipinski_descriptors)
        descriptors = pd.DataFrame(data['descriptors'].tolist())
    
        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)

    # Thermo Descriptors:
    if descriptor_type == 'thermo':
        data['descriptors'] = data['SMILES'].apply(thermo_descriptors)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)
        
    if descriptor_type == 'thermo_dipole':
        data['descriptors'] = data['SMILES'].apply(thermo_dipole)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)
        
    if descriptor_type == 'thermo_surface_tension':
        data['descriptors'] = data['SMILES'].apply(thermo_surface_tension)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)

    if descriptor_type == 'thermo_viscosity':
        data['descriptors'] = data['SMILES'].apply(thermo_viscosity)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)

    if descriptor_type == 'thermo_skin_toxicity':
        data['descriptors'] = data['SMILES'].apply(thermo_skin_toxicity)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)

    if descriptor_type == 'thermo_density':
        data['descriptors'] = data['SMILES'].apply(thermo_density)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)

    if descriptor_type == 'thermo_flash':
        data['descriptors'] = data['SMILES'].apply(thermo_flash)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)

    if descriptor_type == 'thermo_hvap':
        data['descriptors'] = data['SMILES'].apply(thermo_hvap)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)        

    if descriptor_type == 'thermo_melting':
        data['descriptors'] = data['SMILES'].apply(thermo_melting)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)
        
    if descriptor_type == 'thermo_boiling':
        data['descriptors'] = data['SMILES'].apply(thermo_boiling)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)

    if descriptor_type == 'all_thermo':
        data['descriptors'] = data['SMILES'].apply(all_thermo)
        data = data.dropna()
        descriptors = pd.DataFrame(data['descriptors'].tolist())

        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)
        
    # Joback Descriptors:
    if descriptor_type =='joback':
        data['descriptors'] = data['SMILES'].apply(joback_descriptors)
        data = data.dropna()        
        descriptors = pd.DataFrame(data['descriptors'].tolist())
        
        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)    
        
    # Morgan Descriptors:
    if descriptor_type=='morgan':
        data['descriptors'] = data['mol'].apply(morgan_descriptors)
        data = data.dropna() 
        descriptors = pd.DataFrame(data['descriptors'].tolist())
        
        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)   
        
    # RDKit Fingerprint:
    if descriptor_type=='rdkit':
        data['descriptors'] = data['mol'].apply(rdkit_fingerprint)
        data = data.dropna() 
        descriptors = pd.DataFrame(data['descriptors'].tolist())
        
        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None)  
        
        
    # Hansen Solubility Parameters Fingerprint:
    if descriptor_type=='hansen':
        data['descriptors'] = data['mol'].apply(hansen_parameters)
        data = data.dropna() 
        descriptors = pd.DataFrame(data['descriptors'].tolist())
        
        descriptors.to_csv(os.path.join(out_folder,'features.csv'),index=None)
        data[['SMILES','Solubility']].to_csv(os.path.join(out_folder,'data.csv'),index=None) 
        
if __name__ == '__main__':
    generate_starting_files()