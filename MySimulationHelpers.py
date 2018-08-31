#
# Copyright 2018 Manish Kushwaha
#  Contact manish.k79@gmail.com for usage other than personal use.
# 
# MySimulationHelpers.py
#
# Python Deployment Simulation Helper Functions, including,
#  
# 

import math
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

from scipy.optimize import least_squares
import matplotlib.pyplot as plt 
import matplotlib.lines as lines

import MyGeometry as mgeo



def getRangeAndRssiMeas(dlos,dnlos,maxdist):
    """ this function returns the Range and RSSI measurements for a list of LOS/NLOS
        distances between a source and destination. 
        maxdist: maximum multiple range measurement possible 
        The signal processing model used is the time-delayed sinc function,
          y = 1/d**(1/2)  sinc(w(x-d)) + sum_nlos_i{ 1/di**(1/2)  sinc(w(x-di)) }
    
        returns a (range, rssi) where 
         range is the measured range 
         rssi is the measured signal strength  """
    x = np.arange(0, maxdist)
    w = 0.01
    y = np.zeros((x.size))        

    if dlos.size>0:
        if dlos[0]>1e-10:
            y1 = (1./(dlos[0]**(1/2))) * np.sinc(w*(x-dlos[0]))
            y = y + y1
#         print(dlos[0],1./(dlos[0]**(1/2)),x[np.argmax(y)], np.amax(y))
#         plt.plot(x, y1, ls='dashed')
    for d1 in dnlos:
        if d1>1e-10:
            y1 = (1./(d1**(1/2))) * np.sinc(w*(x-d1))
            y = y + y1
#         print(d1,1./(d1**(1/2)),x[np.argmax(y)], np.amax(y))
#         plt.plot(x, y1, ls='dashed')

    rng = -99. if np.amax(y)==0. else x[np.argmax(y)]
    return (rng, np.amax(y))

    
def getNetworkGeometry(tx_csvfilename, obst_csvfilename, fig, verb=1):
    # read the transmitter location file7
    df = pd.read_csv(tx_csvfilename)
    txy = np.array(df.loc[:])

    # read the obstable location file
    df = pd.read_csv(obst_csvfilename)
    xyobs = np.array(df.loc[:])

    if verb>0:
        ax = fig.gca()
        ax.scatter(txy[:,1],txy[:,2],s=80)
        for i, tid in enumerate(txy[:,0]):
            ax.annotate(tid, xy=(txy[i,1], txy[i,2]), xytext=(2,2), textcoords="offset points")
    
        for oid,a,b,c,d in xyobs:
            l1 = lines.Line2D([a, c], [b, d], lw=2, color='black', axes=ax)
            ax.annotate("obs-"+str(oid), xy=((a+c)/2, (b+d)/2))
            ax.add_line(l1)

        ax.grid()
        ax.set_xlim(0,1000)
        ax.set_ylim(0,1000)
        ax.set_aspect('equal')
        #plt.show()
    
    return (txy,xyobs)

    
def getTrajectory(waypoint_csvfilename, fig, speed=10, verb=1, maxdist=1000):
    # read the waypoint location file, and generate path
    df = pd.read_csv(waypoint_csvfilename)
    wxy = np.array(df.loc[:])

    print("DEBUG:: {} input speed/step-size = {!s}".format("getTrajectory", speed))    
    trajId = int(Path(waypoint_csvfilename).stem.rsplit("_")[-1])
    if trajId==99:
        # special case, return meshgrid of step size speed
        x = np.arange(1,maxdist+2,speed)
        y = np.arange(1,maxdist+2,speed)
        X,Y = np.meshgrid(x,y)
        pxy = np.column_stack((X.reshape(-1,1), Y.reshape(-1,1)))
        if verb>0:
            print("Special traj_99...  meshgrid of step-size {!s}, shape {!s}".format(speed,pxy.shape))
    else:
        pxy = mgeo.points2path(wxy[:,1:],speed,verb) # todo: add verbosity input

    if verb>0:
        ax = fig.gca()
        ax.scatter(pxy[:,0],pxy[:,1],s=1,c='green')
        ax.scatter(wxy[:,1],wxy[:,2],s=40,c='red')

        ax.grid()
        ax.set_xlim(0,1000)
        ax.set_ylim(0,1000)
        ax.set_aspect('equal')
        #plt.show()
    
    return pxy

def generateTruthFile(pxy,truth_csv,verb):
    npx = pxy.shape[0]

    truth_data = np.zeros((npx,3))
    truth_data[:,0] = np.arange(0,npx)
    truth_data[:,1:] = pxy

    df = pd.DataFrame(truth_data,columns=['timestamp','x','y'])
    df.to_csv(truth_csv,index=False)
    
def generateMeasurements(txy,pxy,xyobs,meas_csv,maxdist,verb):
    # for all path points, tramistter pairs
    #   find all possible paths, LOS, NLOS (via obstacles in circle region, C((p+t)/2,|p-t|))
    #   scale=1/distance**2, offset=distance/c(=1)
    #   calculate combined signal, sum_i (alp_i * cos(w*(t-tau_i)))
    #   run peak detection on combined signal to estimate, signal strength(magnitude) and range
    #   store the measurements
    ntx = txy.shape[0]
    npx = pxy.shape[0]

    truth_data = np.zeros((npx,3))
    truth_data[:,0] = np.arange(0,npx)
    truth_data[:,1:] = pxy

    meas_data = np.array([])
    
    denom = len(truth_data)*len(txy)

    for ts in truth_data[:,0]:
        px = truth_data[int(ts),1:]
        for tx in txy[:,:]:
            los,nlos = mgeo.findAllPathBetween(tx[1:],px,xyobs[:,1:])
            dlos = np.array([])
            rlos = np.array([np.linalg.norm(px-tx[1:])])
            if los==True:
                dlos = np.array([np.linalg.norm(px-tx[1:])])
            dnlos = np.zeros((nlos.shape[0]))
            for i in range(nlos.shape[0]):
                dnlos[i] = np.linalg.norm(px-nlos[i,:]) + np.linalg.norm(tx[1:]-nlos[i,:])
            if verb>2:
                print("...{!s}   \t-->  {!s}       \tLOC = {}{},   \tNLOS = {!s}{}".format(
                    tx,px,los,dlos,nlos.shape[0],dnlos))

            rng,rssi = getRangeAndRssiMeas(dlos,dnlos,maxdist)

            if meas_data.size==0:
                meas_data = np.array([ts,tx[0],tx[1],tx[2],rng,rssi,los,rlos,nlos.shape[0]])
            else:
                meas_data = np.vstack((meas_data,np.array([ts,tx[0],tx[1],tx[2],rng,rssi,los,rlos,nlos.shape[0]])))
            if verb>0:
                printProgress(float((len(txy)*(ts-1)+tx[0])/denom), "Epoch = {!s}, Tx = {!s}".format(ts,tx))


    df = pd.DataFrame(meas_data,columns=['timestamp','tid','tx','ty','range','rssi','los','dlos','nlos'])
    df.to_csv(meas_csv,index=False)

def printProgress(x, istr):
    print("{} \t\t {:2.1%}".format(istr, x), end="\r")
    
def fun_range_meas(x, *args):
    verb = 0
    data = args[1].reshape(-1,4)
    retf = np.zeros((data.shape[0]))
    if verb>2:
        print(x)
    for i in range(data.shape[0]):
        retf[i] = 100*data[i,3]*data[i,3]*(data[i,2] - np.linalg.norm(data[i,:2]-x))
        if verb>2:
            print(i,data[i,2],data[i,:2],x,retf[i])
    retf.ravel()
    if verb>2:
        print(retf.shape, retf)
    return retf

def jac_range_meas(x, *args):
    verb = 0
    data = args[1].reshape(-1,4)
    retf = np.zeros((data.shape[0],2))
    for i in range(data.shape[0]):
        d = np.linalg.norm(data[i,:2]-x)
        if d<1e-10:
            continue
        retf[i,:] = -100*data[i,3]*data[i,3]*(x-data[i,:2])/d
        #print(i,data[i,:2],x,retf[i,:])
    #print(retf)
    return retf


# Estimate Location Using Generated Measurements 
def makeWlsSolution(meascsvfilename,measoutcsvfilename,poscsvfilename,fig,verb=2):
# todo:: add a progress dot in the loop

    # for all path points (i.e., epochs)
    #   construct LS problem for estimation
    #   construct WLS problem for estimation (weight=f(CN0))
    #   store estimated location, weights, and residuals
    df = pd.read_csv(meascsvfilename)
    df['residual'] = 0.

    nepoch = int(df.iloc[-1].timestamp + 1)
    estpos = np.zeros((nepoch,5))
    svused = np.zeros((nepoch,1),dtype=np.object)
    for t in range(nepoch):
        idx = df[(df['timestamp']==t) & (df['range']>0) & np.isfinite(df['range']) ].index.tolist()
        df2 = df.iloc[idx]
        df2 = df.loc[(df['timestamp']==t) & (df['range']>0) & np.isfinite(df['range'])]# & (df['los']>0) & (df['nlos']<1)]
        if t>-1:
            if len(idx)>0:
                x0 = 10*np.ones((1,2)) + np.array([df2.loc[df2['rssi'].idxmax()].iloc[2:4]])
                x0 = x0[0]
            else:
                x0 = np.array([500.,500.])
        if verb>0:
            printProgress(float(t/nepoch), "Epoch = {!s}".format(t))
        if verb>3:
            print(x0.shape,x0)
            print(len(df2),df2)
        svused[t] = str(list(df2.tid.astype(int)))

        #solveLeastSquareSolution
        data = np.array(df2.iloc[:,2:6]).ravel()
        fun_range_meas(x0,t,data)
        jac_range_meas(x0,t,data)
        
        res_2 = least_squares(fun_range_meas, x0, jac_range_meas, 
                              bounds=([-200, -200], [1200, 1200]), 
                              args=("dummy",data), verbose=0)
        if verb>2:
            print(res_2.x,res_2.cost)
            
        x0 = res_2.x
        estpos[t,:] = [t,x0[0],x0[1],res_2.cost,len(res_2.fun)]
        df.loc[idx,'residual'] = res_2.fun

    dfpos = pd.DataFrame(estpos,columns=['timestamp','x','y','cost','#meas'])
    dfpos['sv_used'] = svused
    
    dfpos.to_csv(poscsvfilename,index=False)
    df.to_csv(measoutcsvfilename,index=False)
    
    if verb>1:
        #fig, ax = plt.subplots()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.scatter(estpos[:,1],estpos[:,2],s=1,c='blue')

        ax.grid()
        ax.set_xlim(0,1000)
        ax.set_ylim(0,1000)
        ax.set_aspect('equal')
        plt.show()
        


def parseWlsDataForMLBiasEst(measoutcsvfilename, poscsvfilename, mldatacsvfilename, mldatanormfcsvfilename, verb=2):
# todo:: add a progress dot in the loop

    dfmeas = pd.read_csv(measoutcsvfilename)
    dfpos = pd.read_csv(poscsvfilename)

    NUM_SV = 20
    NUM_FEATURE_POS = 3
    NUM_FEATURE_PER_SV = 4
    NUM_FEATURE = NUM_FEATURE_POS + NUM_SV * NUM_FEATURE_PER_SV

    col = ['x','y','#meas']
    for i in np.arange(1,NUM_SV+1):
        s = str(i)
        col1 = ['az'+s,'range'+s,'rssi'+s,'res'+s]
        col = np.append(col,col1)

    nepoch = int(dfpos.iloc[-1].timestamp + 1)
    mld = np.zeros((nepoch,NUM_FEATURE),dtype=np.float)
    for t in range(nepoch):
        mld[t,:3] = dfpos.loc[t,['x','y','#meas']]
        starti=NUM_FEATURE_POS
        endi=starti+NUM_FEATURE_PER_SV
        for sid in np.arange(1,NUM_SV+1):
            tpos = dfmeas.loc[(dfmeas.timestamp==t)&(dfmeas.tid==sid),['tx','ty']]
            az = math.atan2(tpos.ty-mld[t,1],tpos.tx-mld[t,0])
            #print(tpos,mld[t,:2],az*180/math.pi)
            #print(dfmeas.loc[(dfmeas.timestamp==t)&(dfmeas.tid==sid),['tx','ty','range','rssi','residual']])
            mld[t,starti] = az
            mld[t,starti+1:endi] = dfmeas.loc[(dfmeas.timestamp==t)&(dfmeas.tid==sid),['range','rssi','residual']]
            starti = endi
            endi = starti+NUM_FEATURE_PER_SV
            if verb>0:
                printProgress(float((NUM_SV*(t-1)+sid)/(nepoch*NUM_SV)), "Epoch = {!s}, Tx = {!s}".format(t,sid))

    df = pd.DataFrame(mld,columns=col)
    df.to_csv(mldatacsvfilename,index=False)
    
    normf = np.zeros((2,NUM_FEATURE),dtype=np.float)
    normf[1,:3] = [1000.,1000.,20.]
    if verb>2:
        print(normf.shape,normf)
    starti=NUM_FEATURE_POS
    endi=starti+NUM_FEATURE_PER_SV
    for sid in np.arange(1,NUM_SV+1):
        normf[1,starti:endi] = [np.pi,1500.,1.,1500.]
        starti = endi
        endi = starti+NUM_FEATURE_PER_SV
        
    df = pd.DataFrame(normf,columns=col)
    df.to_csv(mldatanormfcsvfilename,index=False)

def constructTrainTestMlData(mlCsvFname, mlNormfCsvFname, truthCsvFname,
                              shuffleBeforeTestSplit=False, shuffleTrainData=True,
                              testSplitPc=0.1, testIdx=np.array([]), numFeature=83, verb=2):

    df = pd.read_csv(mlCsvFname)
    nepoch = len(df)

    idx = np.arange(nepoch)
    #allidx = np.arange(nepoch)
    
    if numFeature!=83 and numFeature!=81: 
        print("Other feature lengths " + str(numFeature) + " not defined.. reverting to 83 features")   
        numFeature = 83
        
    if testIdx.size>0:
        # no shuffle before split, split specific indices
        test_idx = idx[testIdx]
        train_idx = np.delete(idx,testIdx)
    else:
        if shuffleBeforeTestSplit:
            print("Shuffling index before train-test split.. ")
            np.random.shuffle(idx)
        if testSplitPc>0.:
            numTest = int(math.floor(nepoch*testSplitPc))
            test_idx = idx[-numTest:]
            train_idx = idx[:-numTest]
        else:
            numTest = 0
            test_idx = np.array([])
            train_idx = idx

    if shuffleTrainData:
        np.random.shuffle(train_idx)
    if verb>2:
        print(test_idx, train_idx)

    if numFeature==83:
        test_data = np.array(df.iloc[test_idx,:])
        train_data = np.array(df.iloc[train_idx,:])
    else:
        test_data = np.array(df.iloc[test_idx,2:])
        train_data = np.array(df.iloc[train_idx,2:])
        
    #all_data = np.array(df.iloc[allidx,:])
    if verb>2:
        print(test_data.shape,train_data.shape)

    dft = pd.read_csv(truthCsvFname)
    truthpos = np.array(dft.loc[:,['x','y']])
    estpos = np.array(df.loc[:,['x','y']])
    posbias = estpos - truthpos
    
    if test_idx.size==0:
        test_label = np.array([])
    else:
        test_label = posbias[test_idx,:]
    train_label = posbias[train_idx,:]
    #all_label = posbias[allidx,:]
    if verb>2:
        print(test_label.shape,train_label.shape)

#     all_data0 = all_data
#     print(allidx[:10])
#     print(all_data0[:10,:3])
    
    # import pickle as pkl
    # pkl.dump([train_data,train_label,test_data,test_label],open('ml_data.pkl','wb'))
    
    # Test data is *not* used when calculating the mean and std.
    df = pd.read_csv(mlNormfCsvFname)

    if numFeature==83:
        mean = np.array(df.iloc[0,:])
        std = np.array(df.iloc[1,:])
    else:
        mean = np.array(df.iloc[0,2:])
        std = np.array(df.iloc[1,2:])
        
    if verb>2:
        print(mean,std)

    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    #all_data = (all_data - mean) / std 

    if verb>2:
        print(train_data[0])  # First training sample, normalized
    
    return (train_data,train_label),(test_data,test_label),(mean,std)

    
def generateRealTimeMeasurements(networkIdx, trajIdx, verb=2, maxdist=2000, speed=10):
    # np.set_printoptions(precision=0)
    
    # Input filenames
    txPosCsvFname = "data\\deployment_{0!s}\\{1}.csv".format(networkIdx, "txpos")
    obsCsvFname = "data\\deployment_{0!s}\\{1}.csv".format(networkIdx, "obstacles")
    trajCsvFname = "data\\deployment_{0!s}\\{2}_{1!s}.csv".format(networkIdx, trajIdx, "waypoints")
    
    print("DEBUG::{} input speed/step-size = {!s}".format("generateRealTimeMeasurements", speed))    
    
    if not Path(txPosCsvFname).is_file():
        print("Transmitter Position file \"{}\" DOES NOT exists!".format(txPosCsvFname))
    if not Path(obsCsvFname).is_file():
        print("Obstable Position file \"{}\" DOES NOT exists!".format(obsCsvFname))
    if not Path(trajCsvFname).is_file():
        print("Trajectory Waypoint file \"{}\" DOES NOT exists!".format(trajCsvFname))

    fig = plt.figure(figsize=(8, 8))
    txy,xyobs = getNetworkGeometry(txPosCsvFname, obsCsvFname, fig, verb)
    pxy = getTrajectory(trajCsvFname, fig, speed=speed, verb=verb)
    
    ax = fig.gca()
    ax.grid()
    plt.show()

    # Output filenames
    measCsvFname = "data\\deployment_{0!s}\\traj_{1!s}\\{2}_{1!s}.csv".format(networkIdx, trajIdx, "meas_data")
    measOutCsvFname = "data\\deployment_{0!s}\\traj_{1!s}\\{2}_{1!s}.csv".format(networkIdx, trajIdx, "meas_data_out")
    truthCsvFname = "data\\deployment_{0!s}\\traj_{1!s}\\{2}_{1!s}.csv".format(networkIdx, trajIdx, "truth")
    estPosCsvFname = "data\\deployment_{0!s}\\traj_{1!s}\\{2}_{1!s}.csv".format(networkIdx, trajIdx, "est_pos")

    if (not Path(truthCsvFname).is_file()):
        # this function takes time.. DO NOT REPEAT
        # Generate Measurements 
        Path(truthCsvFname).parent.mkdir(parents=True, exist_ok=True)
        print("Generating Truth Position file (\'{}\')...".format(truthCsvFname))
        generateTruthFile(pxy, truthCsvFname, 1)
        print("...DONE")
    
    if (not Path(measCsvFname).is_file()):
        # this function takes time.. DO NOT REPEAT
        # Generate Measurements 
        Path(measCsvFname).parent.mkdir(parents=True, exist_ok=True)
        print("Generating Meas Data file (\'{}\')...".format(measCsvFname))
        generateMeasurements(txy, pxy, xyobs, measCsvFname, maxdist, verb)
        print("...DONE")
    else: 
        print("Meas Data file (\'{}\') ALREADY exists...".format(measCsvFname))
    
    #np.set_printoptions(precision=2)
    if (not Path(estPosCsvFname).is_file()):
        # this function takes time.. DO NOT REPEAT
        # Generate Measurements 
        Path(estPosCsvFname).parent.mkdir(parents=True, exist_ok=True)
        print("Generating Meas DataOut file (\'{}\')...".format(measOutCsvFname))
        print("Generating Estimated Position file (\'{}\')...".format(estPosCsvFname))
        makeWlsSolution(measCsvFname, measOutCsvFname, estPosCsvFname, fig, verb=verb)
        print("...DONE")
    else: 
        print("Meas DataOut file (\'{}\') ALREADY exists...".format(measOutCsvFname))
        print("Estimated Position file (\'{}\') ALREADY exists...".format(estPosCsvFname))

    return truthCsvFname,measOutCsvFname,estPosCsvFname

