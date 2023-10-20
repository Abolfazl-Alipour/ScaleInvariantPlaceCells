#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 20:28:05 2022

@author: panlab
"""

# from https://github.com/R-Stefano/Grid-Cells/blob/master/ratSimulator.py

import numpy as np
import matplotlib.pyplot as plt
class RatSimulator():
    def __init__(self, n_steps):
        self.number_steps=n_steps
        self.dt=0.02
        self.maxGap=800
        self.minGap=-800

        self.velScale=130#0.13
        self.mAngVel=0
        self.stddevAngVel=330


    def generateTrajectory(self):
        velocities=np.zeros((self.number_steps))
        angVel=np.zeros((self.number_steps))
        positions=np.zeros((self.number_steps, 2))
        angle=np.zeros((self.number_steps))

        for t in range(self.number_steps):
            #Initialize the agent randomly in the environment
            if(t==0):
                pos=np.array([0,0])#pos=np.random.uniform(low=0, high=2000, size=(2))#
                facAng=0#np.random.uniform(low=-np.pi, high=np.pi)
                prevVel=0

            #Check if the agent is near a wall
            # lastusedT=0
            if(self.checkWallAngle(facAng, pos)):
                    # if t-lastusedT>=600:
                        #if True, compute in which direction turning by 90 deg
                        rotVel=np.deg2rad(np.random.normal(self.mAngVel, self.stddevAngVel))
                        dAngle=self.computeRot(facAng, pos) + rotVel*0.02
                        # breakpoint()
                        # if abs(dAngle-facAng)<.1:
                            # dAngle=0
                        newFacAng=dAngle
                        # lastusedT=t
                    #Velocity reduction factor
                        # vel=np.squeeze(prevVel - (prevVel*0.25))
                    #If the agent is not near a wall, randomly sampling velocity and angVelocity

            else:
                #Sampling velocity
                vel=np.random.rayleigh(self.velScale)
                #Sampling angular velocity
                rotVel=np.deg2rad(np.random.normal(self.mAngVel, self.stddevAngVel))
                dAngle=rotVel*0.02
                newFacAng=(facAng + dAngle)
            #Update the position of the agent
            newPos=pos + (np.asarray([np.cos(facAng), np.sin(facAng)])*vel)*self.dt

            #Update the facing angle of the agent
            # newFacAng=(facAng + dAngle)
            #Keep the orientation between -np.pi and np.pi
            if(np.abs(newFacAng)>=(np.pi)):
                newFacAng=-1*np.sign(newFacAng)*(np.pi - (np.abs(newFacAng)- np.pi))

            velocities[t]=vel
            # angVel[t]=rotVel
            angVel[t]=(dAngle / 0.02)
            positions[t]=pos
            angle[t]=facAng

            pos=newPos
            facAng=newFacAng
            prevVel=vel

        '''
        #USED TO DISPLAY THE TRAJECTORY ONCE FINISHED
        fig=plt.figure(figsize=(15,15))
        ax=fig.add_subplot(111)
        ax.set_title("Trajectory agent")
        ax.plot(positions[:,0], positions[:,1])
        # ax.set_xlim(-2.2,2.2)
        # ax.set_ylim(-2.2,2.2)

        plt.show()
        # '''

        return velocities, angVel, positions, angle

    #HELPING FUNCTIONS
    # for circular boundaries
    # def checkWallAngle(self, ratAng, pos):
    #     #print("Rat orientation:", ratAng)
    #     if((0<=ratAng and ratAng<=(np.pi/2)) and np.all(pos>0)and (np.sqrt(pos[0]**2+pos[1]**2) >self.maxGap ) ):
    #       return True
    #     elif((ratAng>=(np.pi/2) and ratAng<=np.pi) and (pos[0]<0 and pos[1]>0)and (np.sqrt(pos[0]**2+pos[1]**2) >self.maxGap )):
    #       return True
    #     elif((ratAng>=-np.pi and ratAng<=(-np.pi/2)) and np.all(pos>0) and (np.sqrt(pos[0]**2+pos[1]**2) >self.maxGap )):
    #       return True
    #     elif((ratAng>=(-np.pi/2) and ratAng<=0) and (pos[0]>0 and pos[1]<0)and (np.sqrt(pos[0]**2+pos[1]**2) >self.maxGap )):
    #       return True
    #     else:
    #       return False
    def checkWallAngle(self, ratAng, pos):
        # breakpoint()
        # print("Rat orientation:", ratAng)
        if  (np.sqrt(pos[0]**2+pos[1]**2) >self.maxGap ):
        # if((0<=ratAng and ratAng<=(np.pi/2))and np.all(pos>0) and (  (pos[0]**2+pos[1]**2) >self.maxGap)):
            return True
        # elif((ratAng>=(np.pi/2) and ratAng<=np.pi)and (pos[0]<0 and pos[1]>0) and (np.sqrt(pos[0]**2+pos[1]**2) >self.maxGap )):
        #    return True
        # elif((ratAng>=-np.pi and ratAng<=(-np.pi/2))and np.all(pos>0) and (np.sqrt(pos[0]**2+pos[1]**2) >self.maxGap )):
        #    return True
        # elif((ratAng>=(-np.pi/2) and ratAng<=0)and (pos[0]>0 and pos[1]<0) and (np.sqrt(pos[0]**2+pos[1]**2) >self.maxGap)):
        #    return True
        else:
          return False
       # square boundaries
    # def checkWallAngle(self, ratAng, pos):
    #     #print("Rat orientation:", ratAng)
    #     if((0<=ratAng and ratAng<=(np.pi/2)) and np.any(pos>self.maxGap)):
    #       return True
    #     elif((ratAng>=(np.pi/2) and ratAng<=np.pi) and (pos[0]<self.minGap or pos[1]>self.maxGap)):
    #       return True
    #     elif((ratAng>=-np.pi and ratAng<=(-np.pi/2)) and np.any(pos<self.minGap)):
    #       return True
    #     elif((ratAng>=(-np.pi/2) and ratAng<=0) and (pos[0]>self.maxGap or pos[1]<self.minGap)):
    #       return True
    #     else:
    #       return False

# for square box:
    # def computeRot(self,ang, pos):
    #     rot=0
    #     if(ang>=0 and ang<=(np.pi/2)):
    #       if(pos[1]>self.maxGap):
    #         rot=-ang
    #       elif(pos[0]>self.maxGap):
    #         rot=np.pi/2-ang
    #     elif(ang>=(np.pi/2) and ang<=np.pi):
    #       if(pos[1]>self.maxGap):
    #         rot=np.pi-ang
    #       elif(pos[0]<self.minGap):
    #         rot=np.pi/2 -ang
    #     elif(ang>=-np.pi and ang<=(-np.pi/2)):
    #       if(pos[1]<self.minGap):
    #         rot=-np.pi - ang
    #       elif(pos[0]<self.minGap):
    #         rot=-(ang + np.pi/2)
    #     else:
    #       if(pos[1]<self.minGap):
    #         rot=-ang
    #       elif(pos[0]>self.maxGap):
    #         rot=(-np.pi/2) - ang

    #     return rot
# for circular
    def computeRot(self,ang, pos):
        rot=0
        # breakpoint()

       
        if np.all(pos>=0):
            # if(ang>=0 and ang<=(np.pi/2)):
                # if(pos[1]>pos[0]):
                      # rot=np.pi/36#np.pi/2#  # np.pi*5/4 #-ang
                # elif(pos[0]>pos[1]):
                        rot=np.arctan2(-pos[1],-pos[0])+np.pi/3#-ang#np.pi/2-ang
        elif (pos[0]<0 and pos[1]>0):
            # if(ang>=np.pi/2 and ang<=(np.pi)):
                # if(pos[1]<pos[0]):
                    # rot=np.pi/6
                # elif(pos[0]<pos[1]):
                    # rot=np.pi/2 -ang
              # if(pos[1]>pos[0]):
                      rot=np.arctan2(-pos[1],-pos[0])+np.pi/3#-ang #-ang
        elif np.all(pos<0):
            # if(ang>=np.pi and ang<=np.pi*3/2):
                  # breakpoint()
                  
                    # if(pos[1]>pos[0]):
                        # rot=np.pi/36
                    # elif(pos[0]<pos[1]):
                    # rot=-(ang + np.pi/2)
                        rot=np.arctan2(-pos[1],-pos[0])+np.pi/3#-ang #-ang
                  
        elif (pos[0]>0 and pos[1]<0):
            # if(ang>=np.pi*3/2 and ang<=2*np.pi):
                # if(pos[1]<pos[0]):
                    # rot=-np.pi/36#
                # elif (pos[0]<pos[1]):
                  rot= np.arctan2(-pos[1],-pos[0])+np.pi/3#+ang #np.pi/2-ang
        return rot        
        # elif(ang>=(np.pi/2) and ang<=np.pi):
        #   if(pos[1]>abs(pos[0])):
        #     rot=np.pi-ang
        #   elif(abs(pos[0])>pos[1]):
        #     rot=np.pi/2 -ang
        # elif(ang>=-np.pi and ang<=(-np.pi/2)):
        #   if(pos[1]<pos[0]):
        #     rot=-np.pi - ang
        #   elif(pos[0]<pos[1]):
        #     rot=-(ang + np.pi/2)
        # else:
        #   if(abs(pos[1])<pos[0]):
        #     rot=-ang
        #   elif(pos[0]>abs(pos[1])):
        #     rot=(-np.pi/2) - ang

       