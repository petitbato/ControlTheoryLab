import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

#-----------------------------------
def myRound(x, base=5):
    
    """
    Returns a float that is the closest multiple of "base" near "x"
    Based on: https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    
    :x: parameter that is rounded to a multiple of "base"
    :base: "base" parameter (optional: default value is 5)
    
    :return: rounded parameter
    """
    
    return float(base * round(float(x)/base))

#-----------------------------------
def SelectPath_RT(path,time,signal):
    
    """
    The function "SelectPath_RT" needs to be included in a "for or while loop".
    
    :path: dictionary input describing a path in time. Example: path = {0: 0, 5: 1, 50: 2, 80: 3, 100: 3}
    :time: time vector.
    :signal: signal vector that is being constructed using the input "path" and the vector "time".
    
    The function "SelectPath_RT" takes the last element in the vector "time" and, given the input "path", it appends the correct value to the vector "signal".
    """    
    
    for timeKey in path:
        if(time[-1] >= timeKey):
            timeKeyPrevious = timeKey    
    
    value = path[timeKeyPrevious]
    signal.append(value)

#-----------------------------------
def Delay_RT(MV,theta,Ts,MV_Delay,MVInit=0):
    
    """
    The function "Delay_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :theta: delay [s]
    :Ts: sampling period [s]
    :MV_Delay: delayed input vector
    :MVInit: (optional: default value is 0)
    
    The function "Delay_RT" appends a value to the vector "MV_Delay".
    The appended value corresponds to the value in the vector "MV" "theta" seconds ago.
    If "theta" is not a multiple of "Ts", "theta" is replaced by Ts*int(np.ceil(theta/Ts)), i.e. the closest multiple of "Ts" larger than "theta".
    If the value of the vector "input" "theta" seconds ago is not defined, the value "MVInit" is used.
    """
    
    NDelay = int(np.ceil(theta/Ts))
    if NDelay > len(MV)-1:
        MV_Delay.append(MVInit)
    else:    
        MV_Delay.append(MV[-NDelay-1])

#-----------------------------------        
def FO_RT(MV,Kp,T,Ts,PV,PVInit=0,method='EBD'):
    
    """
    The function "FO_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :Kp: process gain
    :T: lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method
    
    The function "FO_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (T != 0):
        K = Ts/T
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            if method == 'EBD':
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*MV[-1])
            elif method == 'EFD':
                PV.append((1-K)*PV[-1] + K*Kp*MV[-2])
            elif method == 'TRAP':
                PV.append((1/(2*T+Ts))*((2*T-Ts)*PV[-1] + Kp*Ts*(MV[-1] + MV[-2])))            
            else:
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*MV[-1])
    else:
        PV.append(Kp*MV[-1])

#-----------------------------------
def FOPDT(MV,Kp,T,theta,Ts,MVInit=0,PVInit=0,method='EBD'):
    
    """
    The function "FOPDT" DOES NOT need to be included in a "for or while loop": this block is for offline use.
    
    :MV: input vector
    :Kp: process gain
    :T: lag time constant [s]
    :theta: delay [s]
    :Ts: sampling period [s]
    :MVInit: (optional: default value is 0)    
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method
        
    :return: simulated FOPDT output vector         
    
    The function "FOPDT" returns the simulated output FOPDT vector from the input vector "MV" and the input parameters.
    """    
    
    MVDelay = []
    MVTemp = []
    PVSim = []    
    
    for i in range(0,len(MV)):
        MVTemp.append(MV[i])
        Delay_RT(MVTemp,theta,Ts,MVDelay,MVInit)
        FO_RT(MVDelay,Kp,T,Ts,PVSim,PVInit,method)
            
    return PVSim

#-----------------------------------
def SOPDT(MV,Kp,T1,T2,theta,Ts,MVInit=0,PVInit=0,method='EBD'):
    
    """
    The function "SOPDT" DOES NOT need to be included in a "for or while loop": this block is for offline use.
    
    :MV: input vector
    :Kp: process gain
    :T1: first (or main) lag time constant [s]
    :T2: second lag time constant [s]    
    :theta: delay [s]
    :Ts: sampling period [s]
    :MVInit: (optional: default value is 0)    
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method
        
    :return: simulated SOPDT output vector         
    
    The function "SOPDT" returns the simulated SOPDT output vector from the input vector "MV" and the input parameters.
    """     
    
    MVDelay = []
    MVTemp = []
    PV1 = []
    PVSim = []    
    
    for i in range(0,len(MV)):
        MVTemp.append(MV[i])
        Delay_RT(MVTemp,theta,Ts,MVDelay,MVInit)
        FO_RT(MVDelay,Kp,T1,Ts,PV1,PVInit,method)
        FO_RT(PV1,1,T2,Ts,PVSim,PVInit,method)
            
    return PVSim

#-----------------------------------
def FOPDT_cost(p,MV,PV,Ts,*args):
    
    """
    :p: parameter vector:
        Kp = p[0]: process gain
        T = p[1]: lag time constant [s]
        theta = p[2]: delay [s]
    :MV: input vector used during experimentation
    :PV: experimental output vector obtained in response to the input vector MV
    :Ts: sampling period [s]
    :args: object, axes and line handles for representing PV and the simulated PV at each function call (optional)
        fig: figure object
        ax1: axes object
        l1: line object for PV
        l2: line object for simulated PV
        
    :return: identification cost with FOPDT model
    
    The function "FOPDT_cost" returns the identification cost, i.e. the sum of the model errors squared.
    The model error is the difference between the experimental output "PV" and the simulated output with a FOPDT model and the parameter set "p".
    
    The assumption is that MVInit and PVInit are zero for the use of Delay_RT and FO_RT in the code.
    The 'EBD' discretisation method is used.
    This assumption is met after a "cleaning operation" after which MV[0] = 0 and PV[0] are 0.
    """     
    
    Kp = p[0]
    T = p[1]
    theta = np.max((0,p[2]))
    
    MVDelay = []
    MVTemp = []
    PVSim = []
    t = []
    
    objective = 0
    
    for i in range(0,len(MV)):
        t.append(i*Ts)
        MVTemp.append(MV[i])
        Delay_RT(MVTemp,theta,Ts,MVDelay)
        FO_RT(MVDelay,Kp,T,Ts,PVSim)
        objective = objective + (PV[i] - PVSim[i])**2      
    
    for fig, ax1, l1, l2 in args:
        l1.set_data(t,PV)
        l2.set_data(t,PVSim)
        ax1.set_xlim(0, t[-1]+1)
        ax1.set_ylim(myRound(np.min(PV),0.5)-0.1, myRound(np.max(PV),0.5)+0.1)        
        # Comment following line otherwise optimisation too slow
        # ax1.text(100, 0.1, 'Kp = {0:3.2f}, T = {1:3.2f}, theta = {2:3.2f}'.format(Kp, T, theta), bbox={'facecolor': 'white'})
        clear_output(wait=True)
        display(fig)
            
    return objective

#-----------------------------------
def SOPDT_cost(p,MV,PV,Ts,*args):
    
    """
    :p: parameter vector:
        Kp = p[0]: process gain
        T1 = p[1]: first or main lag time constant [s]
        T2 = p[2]: second lag time constant [s]    
        theta = p[3]: delay [s]
    :MV: input vector used during experimentation
    :PV: experimental output vector obtained in response to the input vector MV
    :Ts: sampling period [s]
    :args: object, axes and line handles for representing PV and the simulated PV at each function call (optional)
        fig: figure object
        ax1: axes object
        l1: line object for PV
        l2: line object for simulated PV
        
    :return: identification cost with SOPDT model
    
    The function "SOPDT_cost" returns the identification cost, i.e. the sum of the model errors squared.
    The model error is the difference between the experimental output "PV" and the simulated output with a SOPDT model and the parameter set "p".
    
    The assumption is that MVInit and PVInit are zero for the use of Delay_RT and FO_RT in the code.
    The 'EBD' discretisation method is used.
    This assumption is met after a "cleaning operation" after which MV[0] = 0 and PV[0] are 0.
    """    
    
    Kp = p[0]
    T1 = p[1]
    T2 = p[2]
    theta = np.max((0,p[3]))    
    
    MVDelay = []
    MVTemp = []
    PV1 = []
    PVSim = []
    t = []
    
    objective = 0
    
    for i in range(0,len(MV)):
        t.append(i*Ts)
        MVTemp.append(MV[i])
        Delay_RT(MVTemp,theta,Ts,MVDelay)
        FO_RT(MVDelay,Kp,T1,Ts,PV1)
        FO_RT(PV1,1,T2,Ts,PVSim)
        objective = objective + (PV[i] - PVSim[i])**2      
    
    for fig, ax1, l1, l2 in args:
        l1.set_data(t,PV)
        l2.set_data(t,PVSim)
        ax1.set_xlim(0, t[-1]+1)
        ax1.set_ylim(myRound(np.min(PV),0.5)-0.1, myRound(np.max(PV),0.5)+0.1)        
        # Comment following line otherwise optimisation too slow
        # ax1.text(100, 0.1, 'Kp = {0:3.2f}, T1 = {1:3.2f}, T2 = {2:3.2f}, theta = {3:3.2f}'.format(Kp, T1, T2, theta), bbox={'facecolor': 'white'})
        clear_output(wait=True)
        display(fig)
            
    return objective

#-----------------------------------        
class Process:
    
    def __init__(self, parameters):
        
        self.parameters = parameters
        self.parameters['Kp'] = parameters['Kp'] if 'Kp' in parameters else 1.0
        self.parameters['theta'] = parameters['theta'] if 'theta' in parameters else 0.0
        self.parameters['Tlead1'] = parameters['Tlead1'] if 'Tlead1' in parameters else 0.0
        self.parameters['Tlead2'] = parameters['Tlead2'] if 'Tlead2' in parameters else 0.0
        self.parameters['Tlag1'] = parameters['Tlag1'] if 'Tlag1' in parameters else 0.0
        self.parameters['Tlag2'] = parameters['Tlag2'] if 'Tlag2' in parameters else 0.0
        self.parameters['nInt'] = parameters['nInt'] if 'nInt' in parameters else 0        

#-----------------------------------        
def Bode(P,omega,Show=True):
    
    """
    :P: Process as defined by the class "Process".
        Use the following command to define the default process which is simply a unit gain process:
            P = Process({})
        
        A delay, two lead time constants and 2 lag constants can be added.
        
        Use the following commands for a SOPDT process:
            P.parameters['Kp'] = 1.1
            P.parameters['Tlag1'] = 10.0
            P.parameters['Tlag2'] = 2.0
            P.parameters['theta'] = 2.0
        
        Use the following commands for a unit gain Lead-lag process:
            P.parameters['Tlag1'] = 10.0        
            P.parameters['Tlead1'] = 15.0        
        
    :omega: frequency vector (rad/s); generated by a command of the type "omega = np.logspace(-2, 2, 10000)".
    :Show: boolean value (optional: default value = True).
        If Show == True, the Bode diagram is show.
        If Show == False, the Bode diagram is NOT show and the complex vector Ps is returned.
    
    The function "Bode" generates the Bode diagram of the process P
    """     
    
    s = 1j*omega
    
    Ptheta = np.exp(-P.parameters['theta']*s)
    PGain = P.parameters['Kp']*np.ones_like(Ptheta)
    PLag1 = 1/(P.parameters['Tlag1']*s + 1)
    PLag2 = 1/(P.parameters['Tlag2']*s + 1)
    PLead1 = P.parameters['Tlead1']*s + 1
    PLead2 = P.parameters['Tlead2']*s + 1
    
    Ps = np.multiply(Ptheta,PGain)
    Ps = np.multiply(Ps,PLag1)
    Ps = np.multiply(Ps,PLag2)
    Ps = np.multiply(Ps,PLead1)
    Ps = np.multiply(Ps,PLead2)

        
    if Show == True:
        
        fig, (ax_gain, ax_phase) = plt.subplots(2,1)
        fig.set_figheight(12)
        fig.set_figwidth(22)

        # Gain part
        ax_gain.semilogx(omega,20*np.log10(np.abs(Ps)),label=r'$P(s)$')   
        gain_min = np.min(20*np.log10(np.abs(Ps)/5))
        gain_max = np.max(20*np.log10(np.abs(Ps)*5))
        ax_gain.set_xlim([np.min(omega), np.max(omega)])
        ax_gain.set_ylim([gain_min, gain_max])
        ax_gain.set_ylabel('Amplitude' + '\n $|P(j\omega)|$ [dB]')
        ax_gain.set_title('Bode plot of P')
        ax_gain.legend(loc='best')
    
        # Phase part
        ax_phase.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Ps)),label=r'$P(s)$')   
        ax_phase.set_xlim([np.min(omega), np.max(omega)])
        ph_min = np.min((180/np.pi)*np.unwrap(np.angle(Ps))) - 10
        ph_max = np.max((180/np.pi)*np.unwrap(np.angle(Ps))) + 10
        ax_phase.set_ylim([np.max([ph_min, -200]), ph_max])
        ax_phase.set_xlabel(r'Frequency $\omega$ [rad/s]')        
        ax_phase.set_ylabel('Phase' + '\n $\,$'  + r'$\angle P(j\omega)$ [°]')
        ax_phase.legend(loc='best')        

    else:
        return Ps
        