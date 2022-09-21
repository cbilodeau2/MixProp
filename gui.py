# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:28:02 2022

@author: camil
"""

import PySimpleGUI as sg
import time
import subprocess


def run_model(smi1,smi2,molfrac1,T,n_models,num_workers=0):
    args_list = [
        'python C:/Users/camil/Documents/Notebooks/ViscosityModel/gui_wrap.py',
        '--smi1 '+str(smi1),
        '--smi2 '+str(smi2),
        '--molfrac1 '+str(molfrac1),
        '--T '+str(T),
        '--n_models '+str(n_models),
        '--num_workers '+str(num_workers)
        ]
    
    output = subprocess.run(' '.join(args_list),capture_output=True)
    print((output.stdout.split()[-2]).split(b':')[-1])
    pred = float((output.stdout.split()[-2]).split(b':')[-1])
    rel = bool((output.stdout.split()[-1]).split(b':')[-1])
    return pred,rel

def check_result(result):
    try:
        return result.ready()
    except:
        return False

if __name__ == "__main__":

    sg.theme('BluePurple')
    
    layout = [[sg.Text('SMILES 1', size =(20, 1)), sg.Input(key='-SMILES 1-')],
              [sg.Text('SMILES 2', size =(20, 1)), sg.Input(key='-SMILES 2-')],
              [sg.Text('X 1', size =(20, 1)), sg.Input(key='-X 1-')],
              [sg.Text('T (K)', size =(20, 1)), sg.Input(key='-T (K)-')],
              [sg.Text('Number of Models (25 max)', size =(20, 1)), sg.Input(key='-n_models-')],
              [sg.Text('Viscosity Prediction (cP):'), sg.Text(size=(20,1), key='-OUTPUT1-')],
              [sg.Text('Reliable?'), sg.Text(size=(20,1), key='-OUTPUT2-')],
              [sg.Text('Time Elapsed (s):'), sg.Text(size=(20,1), key='-OUTPUT3-')],
              [sg.Button('Submit'),sg.Button('Refresh'),sg.Button('Exit')]]
    
    window = sg.Window('Viscosity Prediction', layout)
    result=[]
    
    while True:  # Event Loop
        event, values = window.read()
        print(event, values)
        
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Submit':
            window['-OUTPUT1-'].update("Thinking...")
            window['-OUTPUT2-'].update(" ")
            window['-OUTPUT3-'].update(" ")
            
            
            
            # Update the "output" text element to be the value of "input" element
            smi1 = values['-SMILES 1-']
            smi2 = values['-SMILES 2-']
            molfrac1 = values['-X 1-']
            T = values['-T (K)-']
            n_models = values['-n_models-']
            
            try:
                n_models = int(n_models)
            except:
                print('Number of models needs to be a number')
                break        
            try:
                T = float(T)
            except:
                print('Temperature needs to be a number')
                break
            
            try:
                molfrac1 = float(molfrac1)
            except:
                print('Mole fraction needs to be a number')
                break
    
            start = time.time()

            return_value = window.perform_long_operation(lambda :run_model(smi1,smi2,molfrac1,T,n_models),
                                          '-END KEY-')
            
        if event == '-END KEY-':
            pred,rel = values[event]
            print(pred,rel)
                           
            if rel:
                rel_out='Yes'
            else:
                rel_out='No'
    
            window['-OUTPUT1-'].update("{:.2f}".format(pred))
            window['-OUTPUT2-'].update(rel_out)
            end = time.time()
            time_elapsed = end-start
            window['-OUTPUT3-'].update("{:.2f}".format(time_elapsed))
            
    
    window.close()