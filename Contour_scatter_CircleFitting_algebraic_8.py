# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:49:54 2022

@author: user
"""
# 生成式: https://steam.oxxostudio.tw/category/python/basic/comprehension.html

import numpy as np
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from scipy.optimize import leastsq
from scipy.interpolate import splprep, splev
from scipy.interpolate import interp1d 

# =============================================================================
# if __name__ == '__main__':
#     N = 100
#     X = np.random.rand(N, 2)
#     y = np.random.rand(N) * 2 - 1
#     sc = plt.scatter(X[:, 0], X[:, 1], vmin=-1, vmax=1, c=y, cmap=cm.seismic)
#     plt.colorbar(sc)
#     plt.show()
# =============================================================================

def read_file(filename):
    # path = os.getcwd()
    # df = pd.read_csv(os.getcwd() +'/'+'slot_test.csv')
    df = pd.read_csv(os.getcwd() +'/'+ filename)
    # print(df)
    # print(df.iloc[:,0])
    # print(df.loc[:,['X']])
    # print(df.loc[:,['Y']])
    # print(df.loc[:,['EPE']])
    
    #因matplot中變數格式需為list, 故用.tolist將Dataframe轉換成list
    X, Y, EPE = df['X'].tolist(), df['Y'].tolist(), df['EPE'].tolist()
    # print('X = ',X)
    # print('Y =',Y)
    # print('EPE =',EPE)
    # plt.title(filename.rstrip()+'_EPE')
    # ax = plt.gca()
    # sc = plt.scatter(X, Y, vmin=-6, vmax=6, c=EPE, cmap=cm.seismic, s=20, alpha=0.7) 
    # plt.colorbar(sc)
    # plt.show()
    return X, Y, EPE



def plot(x,y,EPE, title):
    plt.title(title)
    ax = plt.gca()
    sc = plt.scatter(x, y, vmin=-6, vmax=6, c=EPE, cmap=cm.seismic, s=20, alpha=0.7) 
    plt.colorbar(sc)
    plt.show()

def arrange_point(x,y,n):
    print('len(x) =',len(x))
    print('len(x)+n+1 =', len(x)+n+1)
    # xtt = []
    # ytt = []
    # x2 = []
    # y2 = []
    xtt, ytt, x2, y2 = [[] for i in range(4)]
    for i in range(len(x)):
         xt = []
         yt = []
         for j in range(i-n, i+n+1):
            if j+n+1 <= len(x):
                xt.append(x[j])
                yt.append(y[j])
            else:
                x2 = np.hstack((x,x))
                y2 = np.hstack((y,y))
                xt.append(x2[j])
                yt.append(y2[j])
         print('i =', i)
         print('xt =', xt)
         print('yt =', yt)
         xtt.append(xt)
         ytt.append(yt)
         # print('xtt = ',xtt)
         # print('ytt =', ytt)
    print('len(xtt)= ',len(xtt))
    print('len(x) =',len(x))
    print('len(x)+n =', len(x)+n)
    return xtt, ytt

    


#  start to fit raw data to circle
# == METHOD 1 ==
# method_1 = 'algebraic'

# REF: https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html

def fit_circle (x, y):
    try:
        # coordinates of the barycenter
        x_m = np.mean(x)
        y_m = np.mean(y)
        
        # calculation of the reduced coordinates
        u = x - x_m
        v = y - y_m
        
        # linear system defining the center (uc, vc) in reduced coordinates:
        #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
        #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
        Suv  = sum(u*v)
        Suu  = sum(u**2)
        Svv  = sum(v**2)
        Suuv = sum(u**2 * v)
        Suvv = sum(u * v**2)
        Suuu = sum(u**3)
        Svvv = sum(v**3)
        
        # Solving the linear system
        A = np.array([ [ Suu, Suv ], [Suv, Svv]])
        B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
        uc, vc = np.linalg.solve(A, B)
        
        # Back to original coordinates
        xc_1 = x_m + uc
        yc_1 = y_m + vc 
        # Calculation of all distances from the center (xc_1, yc_1)
        Ri_1      = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
        R_1       = np.mean(Ri_1)
        alph = uc**2 +vc**2 + (Suu + Svv)/len(x)
        radius = np.sqrt(alph)
        R_2 = np.sqrt(alph)
        residu =    np.sqrt(sum((Ri_1 -radius)**2)) #sqrt of summation of del_R**2
        residu_1  = np.sqrt(sum((Ri_1-R_1)**2))
        residu2_1 = np.sqrt(sum((Ri_1**2-R_1**2)**2))
        
        print('fit_center of cicle =', xc_1, yc_1)
        # print('acual_radius =', R_1)
        print('fit_radius =', R_2)
        print('fit_residual = ', residu)
        
        # 加centroid to judge convex or concave curve
        # fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
        # ax.plot(x, y, marker='o', color='g')
        # ax.set_aspect('equal')
        # circle = plt.Circle((xc_1, yc_1), radius = R_2, facecolor = "none", ec = "black")
        # ax.add_patch(circle)
        return xc_1, yc_1, radius, residu
    
    except : 
        # pass
        print(e)
        print('probably it is a line'+'x = ',x ,'y=', y)
        return 'NA','NA','NA','NA'



def write_file(filename, NO, xctr, yctr, EPE, xc, yc, radius, error, INSIDE, radius_eff):
    df = pd.DataFrame(list(zip(NO, xctr, yctr, EPE, xc, yc, radius, error, INSIDE, radius_eff)), 
                      columns =['NO','Xctr', 'Yctr','EPE', 'xc', 'yc', 'radius', 'error', 'INSIDE', 'effective_Radius']) 
    df.to_csv(os.getcwd() +'/'+ filename)
    # df.to_csv(savefile_path+'/'+savefile_name, index=False, mode= 'a')

def main():
    filename = 'slot_test.csv'
    output = 'output_slot.csv'
    
    if os.path.isfile(filename): #check file exist or not
        print('found it!')
        Xctr, Yctr, EPE = read_file(filename) #讀檔產生X,Y, EPE
    else:
        print('could not find the file')
    
    print('Xctr =', Xctr)
    fig, ax = plt.subplots()
    # ax.plot(Xctr, Yctr) #連接vertex
    ax.set_aspect('equal')
    
    # plt.title(filename.rstrip('.csv')+'_EPE')
    ax.set_title(filename.rstrip('.csv')+'_EPE')
    
    # ax = plt.gca()
    sc = plt.scatter(Xctr, Yctr, vmin=-6, vmax=6, c=EPE, cmap=cm.seismic, s=20, alpha=0.7) 
    plt.colorbar(sc)
    plt.show()
    
    # =========interpolate for contour==============================
    
    x = np.array(Xctr)
    y = np.array(Yctr)  
    # append the starting x,y coordinates
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]

    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = splprep([x, y], s=0, k = 1, per=True)

    # evaluate the spline fits for 1000 evenly spaced distance values
    xi, yi = splev(np.linspace(0, 1, 1000), tck)
    
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x, y, 'or')
    ax.plot(xi, yi, ':b')
    plt.show()
    
    
    n = 1 #前後抓n個點算curvature
    Xarg, Yarg = arrange_point(Xctr,Yctr,n) #
    
    print('Xarg0 =', Xarg[0])
    print('Yarg0 =', Yarg[0])

    start = 0
    end = len(Xctr)
    # end = 1
    NO, xc_fit, yc_fit, radius_fit, error_fit, INSIDE, radius_eff = [[] for i in range(7)]
  
    for i in range(start, end):
        NO.append(i)
        xc, yc, radius, fit_error = fit_circle(Xarg[i], Yarg[i])
        print('NO =', NO, 'xc =', xc, 'yc =', yc, 'radius =', radius)
        INSIDE_1, radius_eff_1 = check_if_inPOLY(xc, yc, radius, Xctr, Yctr)
        print('INSIDE_1 =', INSIDE_1, 'radius_eff_1 =', radius_eff_1)
        xc_fit.append(xc)
        yc_fit.append(yc)
        radius_fit.append(radius)
        error_fit.append(fit_error)
        INSIDE.append(INSIDE_1)
        radius_eff.append(radius_eff_1)
    
    # print('Xarg[0], Yarng[0]  =', Xarg[0], Yarg[0])
    # print('xc_fit[0] =', xc_fit[0])
    # print('yc_fit[0] =', yc_fit[0])
    # print('radius_fit[0] =', radius_fit[0])
    # print('radius_fit[0] =', error_fit[0])
    # print('INSIDE[0] =', INSIDE[0])
    # print('radius_eff[0] =', radius_eff[0])
        
    # INSIDE, radius_eff = check_if_inPOLY(xc_fit, yc_fit, radius_fit, Xctr, Yctr)
    
        
    # print('NO =', NO)
    # print('Xctr =', Xctr)
    # print('Yctr =', Yctr)
    # print('xc_fit =', xc_fit)
    # print('yc_fit =', yc_fit)
    # print('radius=', radius)
    # print('error_fit =', error_fit)
    # print('INSIDE=', INSIDE)
    
    N = 26
    fig, ax = plt.subplots()
    ax.set_title(filename.rstrip('.csv')+'_curvature')
    # ax.plot(Xctr, Yctr) #連接vertex1
    ax.set_aspect('equal')
    ax.plot(xi, yi, ':b')
    ax.plot(Xarg[N], Yarg[N], marker='o', color='g')
    ax.plot(xc_fit[N], yc_fit[N], marker='x', color = 'black')
    circle = plt.Circle((xc_fit[N], yc_fit[N]), radius = radius_fit[N], facecolor = "none", ec = "red")
    sc = plt.scatter(Xctr, Yctr, vmin=-200, vmax=200, c=radius_eff, cmap=cm.seismic, s=20, alpha=0.7) 
    plt.colorbar(sc)
    ax.add_patch(circle)
   
    # plt.colorbar(sc)
    # plt.show()
    
    print('len(Xctr)=',len(Xctr))
    print('len(Xarg)=',len(Xarg))
    print('len(xc_fit))=',len(xc_fit))
    print('len(INSIDE) =', len(INSIDE))
    write_file(output ,NO, Xctr, Yctr,EPE, xc_fit, yc_fit, radius_fit, error_fit, INSIDE, radius_eff)
    # write_file(output ,xc_fit, yc_fit, radius_fit, error_fit)
    
main()