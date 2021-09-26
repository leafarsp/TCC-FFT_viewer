import serial
from serial.tools.list_ports import comports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import re



class messages():
    def __init__(self):
        self.message=''
        self.isRunning=True
        self.time=0





def main():
    plt.ion()

    fig, ax = plt.subplots(3, 3)
    columnsFFT = ['Freq','FFTx', 'FFTy', 'FFTz']
    columnsAcc = ['Sample','ACCx', 'ACCy', 'ACCz']

    dataFFT={'Freq':[0],
             'FFTx':[0.],
             'FFTy':[0.],
             'FFTz':[0.]}
    dataAcc = {'Sample':[0],
               'ACCx': [0.],
               'ACCy': [0.],
               'ACCz': [0.]}

    dfAcc = pd.DataFrame(data=dataAcc, columns=columnsAcc, index=dataAcc['Sample'])

    dfFFT = pd.DataFrame(data=dataFFT, columns=columnsFFT, index=dataFFT['Freq'])
    #dfFFT.to_excel('FFT.xlsx')
    #dfAcc.to_excel('Acc.xlsx')



    #exit(1)
    ser = serial.Serial()

    ser.baudrate = 115200
    ser.port = ask_for_port()
    ser.timeout = 1

    if ser.is_open == False:
        ser.open()
    line=""
    i=0
    flagstartFFT = False
    flagstartAcc = False
    #plt.figure()
    j=0
    aq_fr=0.
    aq_fr = 0.
    aq_tau=0.
    if ser.is_open == True:
        while (line != "b'-1\n'"):
            line=ser.readline()
            fields = str(line).replace('b''',''). \
                replace('\'', ''). \
                replace('\\','').\
                replace('n','').split(',')
            #print(fields)
            #print(fields)
            if fields[0] == 'Dados de aquisicao':
                try:
                    aq_fr = float(fields[2])
                except:
                    aq_fr=0.
                try:
                    aq_pr = float(fields[4])
                except:
                    aq_pr = 0.

                try:
                    aq_tau = float(fields[6])
                except:
                    aq_tau = 0.

            if flagstartFFT:
                #print(fields[0])
                if (fields[0]=='Fr'):
                    #print(fields[0])
                    try:
                        freq = float(fields[1])
                    except:
                        freq = 0


                    try:
                        fftx = float(fields[3])
                    except:
                        fftx = 0


                    try:
                        ffty = float(fields[5])
                    except:
                        ffty = 0
                        #print(f'ffty={ffty}')


                    try:
                        fftz = float(fields[7])
                    except:
                        fftz = 0
                        #print(f'fftz={fftz}')


                    dataFFTtemp = {'Freq': [freq],
                                   'FFTx': [fftx],
                                   'FFTy': [ffty],
                                   'FFTz': [fftz]}
                    dfFFTtemp = pd.DataFrame(data=dataFFTtemp, columns=columnsFFT, index=dataFFTtemp['Freq'])
                    #dfFFT = pd.concat([dfFFTtemp, dfFFT])
                    dfFFT = dfFFT.append(dfFFTtemp)

                    dfFFTtemp = None
                    #print(dfFFTtemp)
                    #exit(1)
                    if (int(fields[1])>=255):
                        #flagstartFFT = False
                        dfFFT.dropna(how='any',axis=0,inplace=True)
                        #plt.show()
                        #dfFFT.dropna(how='any',axis=0,inplace=True)
                        #print(dfFFT)
                        dfFFT.to_excel('FFT.xlsx')
                        #print(dfFFT)





                if (fields[0] == 't'):

                    try:
                        sample = float(fields[1])*aq_pr
                    except:
                        sample = 0

                    try:
                        accx = float(fields[3])
                    except:
                        accx = 0

                    try:
                        accy = float(fields[5])
                    except:
                        accy = 0

                    try:
                        accz = float(fields[7])
                    except:
                        accz = 0

                    dataAcctemp = {'Sample': [sample],
                                   'ACCx': [accx],
                                   'ACCy': [accy],
                                   'ACCz': [accz]}
                    dfAcctemp = pd.DataFrame(data=dataAcctemp, columns=columnsAcc, index=dataAcctemp['Sample'])
                    #dfAcc = pd.concat([dfAcctemp, dfAcc])
                    dfAcc = dfAcc.append(dfAcctemp)
                    dfAcctemp = None
                    #print(dfFFTtemp)
                    #exit(1)

                    if (int(fields[1])>=500):
                        #dfAcc.dropna(how='any',axis=0,inplace=True)
                        #print(dfAcc)
                        dfAcc.dropna(how='any',axis=0,inplace=True)
                        flagstartFFT = False
                        plt.clf()
                        fig.suptitle(f'Dados de amostragem:freq:{aq_fr} Per: {aq_pr} Tau: {aq_tau}')
                        plt.subplot(3,3,1)
                        plt.title('Eixo X')
                        plt.plot(dfAcc['ACCx'])

                        plt.subplot(3, 3, 2)
                        plt.title('Eixo Y')
                        plt.plot(dfAcc['ACCy'])

                        plt.subplot(3, 3, 3)
                        plt.title('Eixo Z')
                        plt.plot(dfAcc['ACCz'])

                        plt.subplot(3, 3, 4)
                        #plt.title('FFTx')
                        plt.plot(dfFFT['FFTx'])

                        plt.subplot(3, 3, 5)
                        #plt.title('FFTy')
                        plt.plot(dfFFT['FFTy'])

                        plt.subplot(3, 3, 6)
                        #plt.title('FFTz')
                        plt.plot(dfFFT['FFTz'])



                        spx = np.fft.fft(dfAcc['ACCx'])
                        spModx = np.power(spx.real**2 + spx.imag**2,0.5)
                        spModHalfx = spModx[0:int(len(spModx)/2)]
                        plt.subplot(3, 3, 7)
                        #plt.title('FFTx-P')
                        plt.plot(spModHalfx)

                        spy = np.fft.fft(dfAcc['ACCy'])
                        spMody = np.power(spy.real ** 2 + spy.imag ** 2, 0.5)
                        spModHalfy = spMody[0:int(len(spMody) / 2)]
                        plt.subplot(3, 3, 8)
                        #plt.title('FFTy-P')
                        plt.plot(spModHalfy)

                        spz = np.fft.fft(dfAcc['ACCz'])
                        spModz = np.power(spz.real ** 2 + spz.imag ** 2, 0.5)
                        spModHalfz = spModz[0:int(len(spModz) / 2)]
                        plt.subplot(3, 3, 9)
                        #plt.title('FFTz-P')
                        plt.plot(spModHalfz)

                        #print(spModHalf)
                        #print(spMod)



                        #print(dfFFT)
                        #print(dfAcc)
                        plt.show()
                        plt.savefig(f'Grafs\\Grafs_{j}.png')
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        dfFFT.to_excel(f'FFT\\FFT{j}.xlsx')
                        dfAcc.to_excel(f'ACC\\ACC{j}.xlsx')
                        j+=1
                        dfAcc = pd.DataFrame(data=dataAcc, columns=columnsAcc, index=dataAcc['Sample'])

                        dfFFT = pd.DataFrame(data=dataFFT, columns=columnsFFT, index=dataFFT['Freq'])
                        #buttonPressed = False
                        #while (not buttonPressed):
                           # buttonPressed = plt.waitforbuttonpress()


            #print(fields[0])
            if (fields[0] == 'FFT:'):
                flagstartFFT=True



            #print(line)
            #print(fields)
            i+=1
            #if (i>100000):
                #break

    if ser.is_open == True:
        ser.close()




def ask_for_port():
    """\
    Show a list of ports and ask the user for a choice. To make selection
    easier on systems with long device names, also allow the input of an
    index.
    """
    sys.stderr.write('\n--- Available ports:\n')
    ports = []
    for n, (port, desc, hwid) in enumerate(sorted(comports()), 1):
        sys.stderr.write('--- {:2}: {:20} {!r}\n'.format(n, port, desc))
        ports.append(port)
    while True:
        port = input('--- Enter port index or full name: ')
        try:
            index = int(port) - 1
            if not 0 <= index < len(ports):
                sys.stderr.write('--- Invalid index!\n')
                continue
        except ValueError:
            pass
        else:
            port = ports[index]
        return port



if __name__ == '__main__':
    main()
