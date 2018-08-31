#
# Copyright 2018 Manish Kushwaha
#  Contact manish.k79@gmail.com for usage other than personal use.
# 
# MyGeometry.py
#
# Python Geometry Utility Functions, including,
#  getDistanceFromLine
#  pointOfReflection
#  points2path
#  intersectLines

import math
import numpy as np
    

def getDistanceFromLine(a,b,p):
    """ returns distance of a point(p) from a Line(a,b)
    
        returns dist between point (p) from a Line (a,b).
        returns NaN if points a,b are co-incident.
    """
    
    dx = np.linalg.norm(b-a)
    if dx<1e-6:  return np.nan
        
    n = (b-a)/dx
    dist = np.linalg.norm((a-p)-(np.dot(a-p,n))*n)
    return dist

def pointOfReflection(m1,m2,t,p):
    """ returns point of reflection from point t --> p, reflecting
        on surface/Line (m1,m2)
    
        returns a tuple: (xi, yi, valid, r), where
        (xi, yi) is the point of reflection
        r is the scalar multiple such that (xi,yi) = m1 + r*(m2-m1)
            valid == 0 if there is no reflection (invalid)
            valid == 1 if it has a unique reflection ON the segment  """
    d1 = getDistanceFromLine(m1,m2,t)
    d2 = getDistanceFromLine(m1,m2,p)
    if (d2==0 or d1==0):
        return (0.,0.,0,0.)
    alp = (d1/d2)
    beta = alp**2 - 1
    r = (alp**2)*p - t
    delta = (alp**2)*np.dot(p,p) - np.dot(t,t)

    c = np.array([0,0,0],dtype=float)
    c[0] = beta*(np.dot(m1,m1) + np.dot(m2,m2) - 2*np.dot(m1,m2))
    c[1] = 2*beta*np.dot(m1,m2-m1) - 2*np.dot(r,m2-m1)
    c[2] = beta*np.dot(m1,m1) - 2*np.dot(m1,r) + delta

    t_sol = np.roots(c)

    mu = (m2-m1)/np.linalg.norm(m2-m1)
    nu = (0.55)*(m1+m2)
    nu = (m1-nu)-(np.dot(m1-nu,mu))*mu
    nu = nu/np.linalg.norm(nu)

    valid = 0
    q_ret = np.array([0.,0.])
    x_ret = 0.

    for x_sol in t_sol:
        if (x_sol.dtype==np.complex):
            if (x_sol.imag/x_sol.real)<100.0:
                x_sol = x_sol.real
            else:
                continue
        q = (1-x_sol)*m1 + x_sol*m2
        qtu = (q - t)/np.linalg.norm(q-t)
        pqu = (p - q)/np.linalg.norm(p-q)
        v1 = np.dot(qtu,mu)
        v2 = np.dot(pqu,mu)
        w1 = np.dot(qtu,nu)
        w2 = np.dot(pqu,nu)
        
        if ( (abs(v1)<1e-5 or abs(v2)<1e-5 or (abs(v1-v2)/abs(v1) < 1)) and
             (abs(w1)<1e-5 or abs(w2)<1e-5 or (abs(w1+w2)/abs(w1) < 1)) ):
            x_ret = x_sol
            q_ret = q
            valid = 1

    return (q_ret[0],q_ret[1],valid,x_ret)
    
def points2path(wxy, speed, verb=1):
    """ returns a path from a numpy array of waypoints, and speed
        
        returns a numpy array of points on the interpolated waypoint path """

    t1 = -1
    pathxy = np.array([])
    for w1,w2 in zip(wxy[:-1], wxy[1:]):
        dist = (sum((w2-w1)**2))**(1/2)
        interval = math.floor(dist/speed)
        t2 = t1 + 1 + interval
        #print(w1,w2,dist,interval,t1+1,t2+0.1)
        dt = np.arange(t1+1,t2+0.1)
        dt = (dt - t1 - 1) / (t2 - t1 - 1)
        pxy_ = np.zeros((len(dt),2))
        for i,t in zip(range(len(dt)),dt):
            pxy_[i,:] = (1-t)*w1 + t*w2

        t1 = t2
        print("Append to path array of size " + str(len(pathxy)) + ", new path of size " + str(len(pxy_)))
        if (pathxy.size==0):
            pathxy = pxy_
        else: 
            pathxy = np.append(pathxy, pxy_, axis=0)

    print("\nNumber of points of generated path: " + str(len(pathxy)))
    return pathxy

def intersectLines( a, b, p, q ): 
    """ returns the intersection of Line(a,b) and Line(p,q)
        
        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = a + r*(b-a)
        s is the scalar multiple such that (xi,yi) = p + s*(q-p)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is a + r*(b-a)
    # in component form:
    x1, y1 = a;   x2, y2 = b
    dx1 = x2 - x1;  dy1 = y2 - y1

    # the second line is x + s*(y-x)
    x, y = p;   xB, yB = q;
    dx = xB - x;  dy = yB - y;

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0
    return ( xi, yi, 1, r, s )


def rayIntersectWObstacle(t,p,vobs,verb=1):
    """ this function checks if direct line-of-sight (LOS) ray from source (t)
        to destination (p) has clear path in a field of obstructions (xyobs).
    
        returns a tuple (rayClear, ob, ip) where 
         rayClear True if LOS exists, False otherwise
         ob is obstruction that intersects the LOS path
         ip is the point of intersection  """
    rayClear = True
    ip = np.array([0.,0.])
    ob = np.array([])
    for ob in vobs:
        (_,_, valid, r, s) = intersectLines(ob[:2],ob[2:],t,p)
        if ( valid and r>=0 and r<=1 and s>=0 and s<=1):
            ip = (1-r)*ob[:2] + r*ob[2:]
            rayClear=False
            # todo: what is there are more than one 
            #  obstacles in the direct path
            break
    return (rayClear,ob,ip)
    
def findAllPathBetween(t,p,xyobs,verb=1):
    """ this function finds all paths, i.e., direct line-of-sight (LOS) ray, and 
        all indirect non-line-of-sight (NLOS) rays, from a source (t) to a destination (p)
        in a field of obstructions (xyobs). 
        The NLOS is calculated using only a single-reflection model, i.e., any complex
        ray with reflections off of multiple obstacles that may be present are not considered.
    
        returns a tuple (los,nlosxy) where 
         los True if LOS exists, False otherwise
         nlosxy is numpy array of size (N,2) where N is the number of obstruction causing NLOS """
         
    c = (p+t)/2
    r = np.linalg.norm(p-t)

    if verb>3:
        print("Center {!s}, Radius {!s}".format(c,r))
        fig, ax = plt.subplots()
        ax.scatter(p[0],p[1],s=40,color='red')
        ax.scatter(t[0],t[1],s=40,color='green')

    # find all obstacles in the visible range
    vobs = np.array([])
    for ob in xyobs:
        if ( (np.linalg.norm(ob[:2]-c) < r) or (np.linalg.norm(ob[2:]-c) < r) ):
            if vobs.size==0:
                vobs = ob
            else:
                vobs = np.vstack((vobs,ob))
            if verb>3:
                l1 = lines.Line2D([ob[0],ob[2]],[ob[1],ob[3]],color='black',axes=ax)
                ax.add_line(l1)

    if vobs.size>0 and len(vobs.shape)==1:
        v2 = vobs.copy()
        v2.resize((1,4))
        vobs = v2

    if verb>2:
        print("Number of visible Obstables {!s}, Visible objects {!s}".format(vobs.shape[0],vobs))

    # check if LOS is clear
    los,ob,ip = rayIntersectWObstacle(t,p,vobs,verb)
    if verb>3:
        l1 = lines.Line2D([t[0],ip[0]],[t[1],ip[1]],lw=1,ls='-',color='blue',axes=ax)
        ax.add_line(l1)
        l2 = lines.Line2D([p[0],ip[0]],[p[1],ip[1]],lw=1,ls='dashed',color='blue',axes=ax)
        ax.add_line(l2)
    if verb>2:
        if los==True:
            print(" * LOS present...")
        else: 
            print(" * LOS NOT present, intersects with {!s} at {!s}".format(ob,ip))

    # Check for NLOS paths
    nlosxy = np.array([])
    for i,ob in enumerate(vobs):
        (xi,yi, valid, r) = pointOfReflection(ob[:2],ob[2:],t,p)
        if ( valid and r>=0 and r<=1 ):
            vidx = list(set(range(vobs.shape[0]))-set([i]))
            rayClear,ob1,ip1 = rayIntersectWObstacle(t,np.array([xi,yi]),vobs[vidx,:],verb)
            if rayClear==False:
                if verb>3:
                    l1 = lines.Line2D([t[0],ip1[0]],[t[1],ip1[1]],lw=1,ls='-',color='blue',axes=ax)
                    ax.add_line(l1)
                    l2 = lines.Line2D([xi,ip1[0]],[yi,ip1[1]],lw=1,ls='dashed',color='blue',axes=ax)
                    ax.add_line(l2)
                continue
                #print(" * NLOS NOT present, intersects with " + str(ob) + " at " + str(ip))
            if verb>3:
                l2 = lines.Line2D([t[0], xi], [t[1], yi], lw=1, ls='dashed', color='red', axes=ax)
                ax.add_line(l2)
                
            rayClear,ob1,ip1 = rayIntersectWObstacle(np.array([xi,yi]),p,vobs[vidx,:],verb)
            if rayClear==False:
                if verb>3:
                    l1 = lines.Line2D([xi,ip1[0]],[yi,ip1[1]],lw=1,ls='-',color='blue',axes=ax)
                    ax.add_line(l1)
                    l2 = lines.Line2D([p[0],ip1[0]],[p[1],ip1[1]],lw=1,ls='dashed',color='blue',axes=ax)
                    ax.add_line(l2)
                continue
                #print(" * NLOS NOT present, intersects with " + str(ob) + " at " + str(ip))
            if verb>3:
                l2 = lines.Line2D([p[0], xi], [p[1], yi], lw=1, ls='dashed', color='green', axes=ax)
                ax.add_line(l2)
            
            if nlosxy.size==0:
                nlosxy = np.array([xi,yi])
            else:
                nlosxy = np.vstack((nlosxy,np.array([xi,yi])))
            if verb>2:
                print(" * NLOS present with reflection from {!s} at {!s}".format(ob,[xi,yi]))

    if verb>3:
        ax.grid()
        ax.set_xlim(0,1000)
        ax.set_ylim(0,1000)
        plt.show()

    if nlosxy.size>0 and len(nlosxy.shape)==1:
        v2 = nlosxy.copy()
        v2.resize((1,2))
        nlosxy = v2
        
    return (los,nlosxy)
        