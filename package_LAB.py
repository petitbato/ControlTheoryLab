import numpy as np
import package_DBR

from importlib import reload
package_DBR = reload(package_DBR)

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

#-----------------------------------

def LL_RT(MV,Kp,Tlead,Tlag,Ts,PV,PVInit=0,method='EBD'):
    
    """
    The function "LL_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :Kp: process gain
    :Tlead: lead time constant [s]
    :Tlag: lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method
    
    The function "LL_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (Tlag != 0):
        K = Ts/Tlag
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            if method == 'EBD':
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*(((1+(Tlead/Ts))*MV[-1]-(Tlead/Ts)*MV[-2])))
            elif method == 'EFD':
                PV.append((1-K)*PV[-1] + K*Kp*(MV[-1]*Tlead/Ts+(1-Tlead/Ts)*MV[-2]))
            elif method == 'TRAP':
                #PV.append((1/(2*T+Ts))*((2*T-Ts)*PV[-1] + Kp*Ts*(MV[-1] + MV[-2])))
                pass
            else:
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*MV[-1])
    else:
        PV.append(Kp*MV[-1])

#-----------------------------------  

def Proportional_Action(MVP, Kc, E, method):
    if not MVP:
        print("MVP Empty")
        MVP.append(Kc*E[-1])
    else:
        if method == "EBD-EBD":
            MVP.append(Kc*E[-1])
    return MVP
    
def Integral_Action(MVI, Kc, Ts, Ti, E, method):
    if not MVI:
        print("MVI Empty")
        MVI.append(((Kc*Ts)/Ti)*(E[-1]))
    else:
        if method == "EBD-EBD":
            MVI.append(MVI[-1] + ((Kc*Ts)/Ti)*(E[-1]))
    return MVI

def Derivative_Action(MVD, Kc, Tfd, Ts, Td, E, method):
    if not MVD :
        print("MVD Empty")
        MVD.append((Kc*Td/(Tfd+Ts)*E[-1]))
    else:
        if method == "EBD-EBD":
            MVD.append(MVD[-1]*(Tfd/(Tfd+Ts))+(Kc*Td/(Tfd+Ts)*(E[-1]-E[-2])))
    return MVD


def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, method='EBD-EBD'):
    """
    :SP: SP (or SetPoint) vector
    :PV: PV (or Process Value) vector
    :Man: Man (or Manual controller mode) vector (True or False)
    :MVMan: MVMan (or Manual value for MV) vector
    :MVFF: MVFF (or Feedforward) vector
    :Kc: controller gain
    Ti: integral time constant [s]
    Td: derivative time constant [s]
    :alpha: Tfd alpha*Td where Tfd is the derivative filter time constant [s]
    :Ts: sampling period [s]
    :MVMin: minimum value for MV (used for saturation and anti wind-up) :MVMax: maximum value for MV (used for saturation and anti wind-up)
    :MV: MV (or Manipulated Value) vector
    :MVP: MVP (or Propotional part of MV) vector
    :MVI: MVI (or Integral part of MV) vector
    :MVD: MVD (or Derivative part of MV) vector
    :E: E (or control Error) vector
    :ManFF: Activated FF in manual mode (optional: default boolean value is False)
    :PVInit: Initial value for PV (optional: default value is 0): used if PID RT is ran first in the squence and no value of PV is available yet.
    :method: discretisation method (optional: default value is 'EBD') EBD-EBD: EBD for integral action and EBD for derivative action EBD-TRAP: EBD for integral action and TRAP for derivative action TRAP-EBD: TRAP for integral action and EBD for derivative action TRAP-TRAP: TRAP for integral action and TRAP for derivative action
    The function "PID_RT" appends new values to the vectors "MV", "MVP", "MVI", and "MVD".
    The appended values are based on the PID algorithm, the controller mode, and feedforward.
    Note that saturation of "MV" within the limits [MVMin MVMax] is implemented with anti wind-up.
    """ 
    
    Tfd = alpha*Td

    if len(PV) == 0:
        E.append(SP[-1] - PVInit)
    else: 
        E.append(SP[-1] - PV[-1])
    
    # Calcul des valeurs
    
    MVP = Proportional_Action(MVP, Kc, E, method)
    MVI = Integral_Action(MVI, Kc, Ts, Ti, E, method)
    MVD = Derivative_Action(MVD, Kc, Tfd, Ts, Td, E, method)

    # MAN mode
    
    if Man[-1]:
        if ManFF:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1]
            
        else:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFF[-1]

    # Saturation
    if ((MVP[-1] + MVD[-1] + MVFF[-1] + MVI[-1]) > MVMax):
        MVI[-1] = MVMax - (MVP[-1] + MVD[-1] + MVFF[-1])
    if ((MVP[-1] + MVD[-1] + MVFF[-1] + MVI[-1]) < MVMin): 
        MVI[-1] = MVMin - (MVP[-1] + MVD[-1] + MVFF[-1])

        # Résultat
    MV.append(MVP[-1] + MVD[-1] + MVFF[-1] + MVI[-1])

#---------------------------------  
def IMCTuning(K, Tlag1, Tlag2=0, theta=0, gamma = 0.5):
    """
    IMCTuning computes the IMC PID tuning parameters for FOPDT and SOPDT processes.
    K: process gain (Kp)
    Tlag1: first (main) lag time constant [s]
    Tlag2: second lag time constant [s]
    theta: delay [s]

    """
    Tclp = gamma*Tlag1
    Kc = ((Tlag1 + Tlag2)/(Tclp + theta))/K
    Ti = (Tlag1 +Tlag2)
    Td = ((Tlag1*Tlag2))/(Tlag1+Tlag2)
    return Kc, Ti, Td


#-----------------------------------

# Calcul des marges de gain et de phase et affichage des résultats
def Margin_gain_phase(Ps, C, omega, Show=True):
    """
    Calcule et affiche la marge de gain et la marge de phase.
    """
    # Initialisation des paramètres
    s = 1j * omega
    Kc = C.parameters['Kc']
    Ti = C.parameters['Ti']
    Td = C.parameters['Td']
    Tfd = C.parameters['Tfd']

    # Calcul de la fonction de transfert du contrôleur
    Cs = Kc * (1 + 1/(Ti * s) + (Td * s) / (Tfd * s + 1))

    # Calcul de la fonction de transfert en boucle ouverte L(s) = P(s)C(s)
    Ls = Cs * Ps 

    # Tracé de L(s)
    if Show:
        fig, (ax_freq, ax_time) = plt.subplots(2, 1)
        fig.set_figheight(12)
        fig.set_figwidth(22)

        # Amplitude
        ax_freq.semilogx(omega, 20 * np.log10(np.abs(Ls)), label='L(s)')
        gain_min = np.min(20 * np.log10(np.abs(Ls) / 5))
        gain_max = np.max(20 * np.log10(np.abs(Ls) * 5))
        ax_freq.set_xlim([np.min(omega), np.max(omega)])
        ax_freq.set_ylim([gain_min, gain_max])
        ax_freq.set_ylabel('Amplitude |P| [db]')
        ax_freq.set_title('Bode plot of P')
        ax_freq.legend(loc='best')
        
        # Phase
        ax_time.semilogx(omega, (180 / np.pi) * np.unwrap(np.angle(Ls)), label='L(s)')
        ax_time.set_xlim([np.min(omega), np.max(omega)])
        ph_min = np.min((180 / np.pi) * np.unwrap(np.angle(Ps))) - 10
        ph_max = np.max((180 / np.pi) * np.unwrap(np.angle(Ps))) + 10
        ax_time.set_ylim([np.max([ph_min, -200]), ph_max])
        ax_time.set_ylabel(r'Phase $\angle P$ [°]')
        ax_time.legend(loc='best')
        ax_freq.axhline(y=0, color='black')
        ax_time.axhline(y=-180, color='black')

    # Fréquence de croisement
    crossover_frequency = None
    for i, value in enumerate(Ls):
        dB = 20 * np.log10(np.abs(value))
        if dB < 0.05 and dB > -0.05:
            crossover_frequency = omega[i]
            crossover_phase = np.angle(value, deg=True)
            break

    # Fréquence ultime
    ultimate_frequency = None
    for j, value in enumerate(Ls):
        deg = np.angle(value, deg=True)
        if deg < -179.5 and deg > -180.5:
            ultimate_frequency = omega[j]
            ultimate_gain = 20 * np.log10(np.abs(value))
            break

    # Affichage des résultats
    if Show:
        if crossover_frequency is not None and ultimate_frequency is not None:
            ax_freq.plot([ultimate_frequency, ultimate_frequency], [0, ultimate_gain], color='green') 
            ax_time.plot([crossover_frequency, crossover_frequency], [crossover_phase, -180], color='green')
            plt.show()
    if crossover_frequency is not None and ultimate_frequency is not None:
        print(f"Gain margin: {-ultimate_gain} dB at the ultimate frequency: {ultimate_frequency} rad/s")
        print(f"Phase margin: {crossover_phase + 180}° at the crossover frequency: {crossover_frequency} rad/s")
    else:
        print("There are errors in values of crossover frequency and ultimate frequency.")

