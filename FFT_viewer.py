import serial
from serial.tools.list_ports import comports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from datetime import datetime
import re
import os
ROOT_DIR = os.path.abspath(os.curdir)




class messages():
    def __init__(self):
        self.message=''
        self.isRunning=True
        self.time=0





def main():
    plt.ion()

    fig, ax = plt.subplots(4, 3)
    columnsFFT = ['Freq','FFTx', 'FFTy', 'FFTz']
    columnsAcc = ['Sample','ACCx', 'ACCy', 'ACCz']
    columnsSpeed = ['Sample', 'Speed_x', 'Speed_y', 'Speed_z']

    dataFFT={'Freq':[0],
             'FFTx':[0.],
             'FFTy':[0.],
             'FFTz':[0.]}
    dataAcc = {'Sample':[0],
               'ACCx': [0.],
               'ACCy': [0.],
               'ACCz': [0.]}
    dataSpeed = {'Sample': [0],
               'Speed_x': [0.],
               'Speed_y': [0.],
               'Speed_z': [0.]}

    dfAcc = pd.DataFrame(data=dataAcc, columns=columnsAcc, index=dataAcc['Sample'])

    dfFFT = pd.DataFrame(data=dataFFT, columns=columnsFFT, index=dataFFT['Freq'])

    dfSpeed = pd.DataFrame(data=dataSpeed, columns=columnsSpeed, index=dataSpeed['Sample'])
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
    cont_fr_samples = 0
    if ser.is_open == True:
        while (line != "b'-1\n'"):
            line=ser.readline()
            fields = str(line).replace('b''',''). \
                replace('\'', ''). \
                replace('\\','').\
                replace('n','').split(',')
            print(fields)
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

                try:
                    aq_amostras = int(fields[8])
                except:
                    aq_amostras = 0

                try:
                    aq_FFT_fr_res = float(fields[10])

                except:
                    aq_FFT_fr_res = 0

            if fields[0] == 'RMS-ACC':
                try:
                    rms_acc_x = float(fields[2])
                except:
                    rms_acc_x=0.
                try:
                    rms_acc_y = float(fields[4])
                except:
                    rms_acc_y = 0.

                try:
                    rms_acc_z = float(fields[6])
                except:
                    rms_acc_z = 0.

            if fields[0] == 'RMS-Speed':
                try:
                    rms_speed_x = float(fields[2])
                except:
                    rms_speed_x=0.
                try:
                    rms_speed_y = float(fields[4])
                except:
                    rms_speed_y = 0.

                try:
                    rms_speed_z = float(fields[6])
                except:
                    rms_speed_z = 0.

            if fields[0] == 'Pico-ACC':
                try:
                    pico_acc_x = float(fields[2])/1000.
                except:
                    pico_acc_x=0.
                try:
                    pico_acc_y = float(fields[4])/1000.
                except:
                    pico_acc_y = 0.

                try:
                    pico_acc_z = float(fields[6])/1000.
                except:
                    pico_acc_z = 0.

            if flagstartFFT:
                #print(fields[0])
                if (fields[0]=='Fr'):
                    #print(fields[0])
                    cont_fr_samples += 1
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
                    if (cont_fr_samples >=(255)):
                        cont_fr_samples = 0
                        #flagstartFFT = False
                        dfFFT.dropna(how='any',axis=0,inplace=True)
                        #plt.show()
                        #dfFFT.dropna(how='any',axis=0,inplace=True)
                        #print(dfFFT)
                        #dfFFT.to_excel('FFT.xlsx')
                        #print(dfFFT)







                if (fields[0] == 't_a'):
                    #print('t_a\n')
                    try:
                        sample = float(fields[1]) * aq_pr
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
                    # dfAcc = pd.concat([dfAcctemp, dfAcc])
                    dfAcc = dfAcc.append(dfAcctemp)
                    dfAcctemp = None
                    # print(dfFFTtemp)
                    # exit(1)


                if (fields[0] == 't_s'):
                    #print('t_s\n')
                    try:
                        sample = float(fields[1]) * aq_pr
                    except:
                        sample = 0

                    try:
                        speed_x = float(fields[3])
                    except:
                        speed_x = 0


                    try:
                        speed_y = float(fields[5])
                    except:
                        speed_y = 0

                    try:
                        speed_z = float(fields[7])
                    except:
                        speed_z = 0

                    dataSpeedtemp = {'Sample': [sample],
                                     'Speed_x': [speed_x],
                                     'Speed_y': [speed_y],
                                     'Speed_z': [speed_z]}
                    dfSpeedtemp = pd.DataFrame(data=dataSpeedtemp, columns=columnsSpeed,
                                               index=dataSpeedtemp['Sample'])
                    # dfAcc = pd.concat([dfAcctemp, dfAcc])
                    dfSpeed = dfSpeed.append(dfSpeedtemp)
                    dfSpeedtemp = None
                    # print(dfFFTtemp)
                    # exit(1)


                    if (fields[0] == 't_s' and int(fields[1])>=500):
                        print(f't_s: {fields[1]}\n')

                        dfAcc.drop(axis=0,index=0,inplace=True)
                        dfFFT.drop(axis=0,index=0,inplace=True)
                        dfSpeed.drop(axis=0,index=0,inplace=True)

                        subplot_col = 3
                        subplot_lin = 4

                        #dfAcc.dropna(how='any',axis=0,inplace=True)
                        #print(dfAcc)
                        dfAcc.dropna(how='any',axis=0,inplace=True)
                        flagstartFFT = False
                        plt.clf()
                        fig.suptitle(f'freq:{aq_fr:.2f}Hz Per: {aq_pr*1000:.3f}ms Tau: {aq_tau:.3}s '
                                     f'Amostras: {aq_amostras} fft_res:{aq_FFT_fr_res}Hz\n'
                                     f'RMS_vel[mm/s]:x={rms_speed_x:.2f} y={rms_speed_y:.2f} z={rms_speed_z:.2}  '
                                     f'RMS_acc[m/s2]:x={rms_acc_x:.2f} y={rms_acc_y:.2f} z={rms_acc_z:.2}  '
                                     f'Pico_acc[m/s2]:x={pico_acc_x:.2f} y={pico_acc_y:.2f} z={pico_acc_z:.2}')
                        plt.subplot(subplot_lin,subplot_col,1)

                        rms_acc_x_plot = np.zeros(dfAcc['ACCx'].index.size)
                        rms_acc_x_plot.fill(rms_acc_x)



                        rms_acc_y_plot = np.zeros(dfAcc['ACCy'].index.size)
                        rms_acc_y_plot.fill(rms_acc_y)

                        rms_acc_z_plot = np.zeros(dfAcc['ACCz'].index.size)
                        rms_acc_z_plot.fill(rms_acc_z)


                        rms_speed_x_plot = np.zeros(dfSpeed['Speed_x'].index.size)
                        rms_speed_x_plot.fill(rms_speed_x)

                        rms_speed_y_plot = np.zeros(dfSpeed['Speed_y'].index.size)
                        rms_speed_y_plot.fill(rms_speed_y)

                        rms_speed_z_plot = np.zeros(dfSpeed['Speed_z'].index.size)
                        rms_speed_z_plot.fill(rms_speed_z)





                        pico_acc_x_plot = np.zeros(dfAcc['ACCx'].index.size)
                        pico_acc_x_plot.fill(pico_acc_x)

                        pico_acc_y_plot = np.zeros(dfAcc['ACCy'].index.size)
                        pico_acc_y_plot.fill(pico_acc_y)

                        pico_acc_z_plot = np.zeros(dfAcc['ACCz'].index.size)
                        pico_acc_z_plot.fill(pico_acc_z)




                        #print(rms_speed_x_plot)


                        plt.plot(dfAcc['ACCx'])
                        plt.plot(dfAcc['ACCx'].index,rms_acc_x_plot)
                        #plt.plot(pico_acc_x_plot / 100)

                        plt.subplot(subplot_lin, subplot_col, 2)

                        plt.plot(dfAcc['ACCy'])
                        plt.plot(dfAcc['ACCy'].index,rms_acc_y_plot)
                        #plt.plot(pico_acc_y_plot/100)

                        plt.subplot(subplot_lin, subplot_col, 3)

                        plt.plot(dfAcc['ACCz'])
                        plt.plot(dfAcc['ACCz'].index,rms_acc_z_plot)
                        #plt.plot(pico_acc_z_plot / 100)

                        plt.subplot(subplot_lin, subplot_col, 4)
                        plt.plot(dfSpeed['Speed_x'])
                        plt.plot(dfSpeed['Speed_x'].index,rms_speed_x_plot)


                        plt.subplot(subplot_lin, subplot_col, 5)
                        plt.plot(dfSpeed['Speed_y'])
                        plt.plot(dfSpeed['Speed_y'].index,rms_speed_y_plot)

                        plt.subplot(subplot_lin, subplot_col, 6)
                        plt.plot(dfSpeed['Speed_z'])
                        plt.plot(dfSpeed['Speed_z'].index,rms_speed_z_plot)


                        plt.subplot(subplot_lin, subplot_col, 7)
                        #plt.title('FFTx')
                        plt.plot(dfFFT['FFTx'])

                        plt.subplot(subplot_lin, subplot_col, 8)
                        #plt.title('FFTy')
                        plt.plot(dfFFT['FFTy'])

                        plt.subplot(subplot_lin, subplot_col, 9)
                        #plt.title('FFTz')
                        plt.plot(dfFFT['FFTz'])

                        qt_amostras = (dfAcc['Sample']).count()
                        T_amostragem = max(dfAcc['Sample'])

                        fft_beams = 512

                        f_res = (qt_amostras / T_amostragem) / fft_beams
                        #print(f'frequency resolution = {f_res}Hz')

                        f_range = np.linspace(0, (fft_beams / 2) * f_res, int(fft_beams / 2))

                        #spx = np.fft.fft(dfSpeed['Speed_x'],n=fft_beams)
                        #spy = np.fft.fft(dfSpeed['Speed_y'],n=fft_beams)
                        #spz = np.fft.fft(dfSpeed['Speed_z'],n=fft_beams)

                        spx = np.fft.fft(dfAcc['ACCx'], n=fft_beams)
                        spy = np.fft.fft(dfAcc['ACCy'], n=fft_beams)
                        spz = np.fft.fft(dfAcc['ACCz'], n=fft_beams)

                        spModx = np.abs(spx)
                        spModHalfx = spModx[0:int(len(spModx)/2)]
                        spModHalfx = spModHalfx / (qt_amostras/4)

                        plt.subplot(subplot_lin, subplot_col, 10)
                        #plt.title('FFTx-P')
                        print(f'f_range={len(f_range)},spModHalfx={len(spModHalfx)}')
                        plt.plot(f_range,spModHalfx)
                        plt.xlabel('Eixo X')


                        spMody = np.abs(spy)
                        spModHalfy = spMody[0:int(len(spMody) / 2)]
                        spModHalfy = spModHalfy / (qt_amostras/4)
                        plt.subplot(subplot_lin, subplot_col, 11)
                        #plt.title('FFTy-P')
                        print(f'f_range={len(f_range)},spModHalfx={len(spModHalfy)}')
                        plt.plot(f_range, spModHalfy)
                        plt.xlabel('Eixo Y')




                        spModz = np.abs(spz)
                        spModHalfz = spModz[0:int(len(spModz) / 2)]
                        spModHalfz = spModHalfz / (qt_amostras/4)
                        plt.subplot(subplot_lin, subplot_col, 12)
                        #plt.title('FFTz-P')
                        plt.plot(f_range,spModHalfz)
                        plt.xlabel('Eixo Z')

                        """"Verificação do cálculo de velocidade feito pela placa
                        #speed_x_calc = dfAcc['ACCx']
                        df_speed_x = integrate_acc_numerically(dfAcc['ACCx'], 'ACCx', 'Speed_x')
                        plt.subplot(subplot_lin, subplot_col, 13)
                        plt.plot(df_speed_x['Speed_x'])
                        df_speed_y = integrate_acc_numerically(dfAcc['ACCy'], 'ACCy', 'Speed_y')
                        plt.subplot(subplot_lin, subplot_col, 14)
                        plt.plot(df_speed_y['Speed_y'])
                        df_speed_z = integrate_acc_numerically(dfAcc['ACCz'], 'ACCz', 'Speed_z')
                        plt.subplot(subplot_lin, subplot_col, 15)
                        plt.plot(df_speed_z['Speed_z'])
                        """
                        #print(spModHalf)
                        #print(spMod)

                        time_stamp = f'{datetime.now().year}-{datetime.now().month}-{datetime.now().day}_' \
                                     f'{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}_a{j}'
                        #print(dfFFT)
                        #print(dfAcc)


                        plt.savefig(f'{ROOT_DIR}\\Grafs\\Grafs_{time_stamp}.png')
                        dfFFT.to_excel(f'{ROOT_DIR}\\FFT\\FFT_{time_stamp}.xlsx')

                        dfSpeed.to_excel(f'{ROOT_DIR}\\Speed\\Speed_{time_stamp}.xlsx')

                        dfAcc.to_excel(f'{ROOT_DIR}\\ACC\\ACC_{time_stamp}.xlsx')
                        plt.show()

                        fig.canvas.draw()
                        fig.canvas.flush_events()





                        j+=1
                        dfAcc = pd.DataFrame(data=dataAcc, columns=columnsAcc, index=dataAcc['Sample'])
                        dfSpeed = pd.DataFrame(data=dataSpeed, columns=columnsSpeed, index=dataSpeed['Sample'])
                        dfFFT = pd.DataFrame(data=dataFFT, columns=columnsFFT, index=dataFFT['Freq'])
                        #buttonPressed = False
                        #while (not buttonPressed):
                           # buttonPressed = plt.waitforbuttonpress()


            #print(fields[0])
            if (fields[0] == 'FFT:'):
                flagstartFFT=True
                cont_fr_samples = 0



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


def integrate_acc_numerically(df, eixo_acc, eixo_speed):
    ret_df = pd.DataFrame(data = None,index = df.index, columns = ['t_s',eixo_speed])
    ret_df.iloc[0][1]=0

    for i in range(1, df.index.size):
        #print(f'i={i} df.iloc[i]={df.iloc[i]} df.iloc[i-1]={df.iloc[i-1]} df.index[i]={df.index[i]} '
              #f'df.index[i-1]={df.index[i-1]} ret_df.iloc[i][1]={ret_df.iloc[i-1][1]}')
        dt = (df.index[i]-df.index[i-1])
        ret_df.iloc[i] = (((df.iloc[i]+df.iloc[i-1])/2) * dt)*1000. + ret_df.iloc[i-1][1]


    ret_df.iloc[0][1] = ret_df.iloc[1][1]
    ret_df[eixo_speed] = ret_df[eixo_speed] - ret_df[eixo_speed].mean()
    return ret_df



if __name__ == '__main__':
    main()
