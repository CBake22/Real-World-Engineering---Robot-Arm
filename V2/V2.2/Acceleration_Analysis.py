# This code provides an example acceleration analysis of a robot manipulator

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Code from EML 6282, Reverse acceleration analysis
# for T3-776 industrial manipulator.


# In[2]:


import numpy as np
import math
D2R = math.pi/ 180.0 
R2D = 180.0/math.pi
np.set_printoptions(precision = 3)


# In[3]:


# input items - constant mechanism parameters
a1 = 0	# inches
a2 = 44.0
a3 = 0
a4 = 0
a5 = 0
alphal	= 90.0*D2R
alpha2	=  0.0*D2R
alpha3	= 90.0*D2R
alpha4	= 61.0*D2R
alpha5	= 61.0*D2R
S_2t01	=  0.0	# inches
S_3t02	=  0.0
S_4t03	= 55.0
S_5t04	=  0.0
# input items - joint angles 
phi_1to0 =  37.0*D2R
th_2t01	 =  85.0*D2R
th_3t02	 = -23.0*D2R
th_4t03	 =  71.0*D2R
th_5t04	 = 127.0*D2R
th_6t05	 = 101.0*D2R
#input items - user selected values
S_6t05	 = 6.0 #inches
#input items - desired velocity state
P_tool_in6 = np.array([5.0, 3.0, 7.0]) # inches
V_tool_inF = np.array([2.0, 4.0, -7.0]) # inches/sec 
omega_6to0 = np.array([8.0*D2R, 0.0, 0.0]) # rad/sec


# In[4]:


# define a function to Load transformation matrices
def load_T(a_i, alpha_i, theta_ij, S_ij ):
    sij = math.sin(theta_ij )
    cij = math.cos(theta_ij )
    si  = math.sin(alpha_i)
    ci  = math.cos(alpha_i)
    T_mat = np.array([[cij, -sij, 0, a_i],\
                      [sij*ci, cij*ci, -si, -si*S_ij],\
                      [sij*si, cij*si, ci, ci*S_ij],\
                      [0,0,0,1]])
    return T_mat


# In[5]:


def load_TF(phi_01):
    s_phi = math.sin(phi_01)
    c_phi = math.cos(phi_01)

    T_mat = np.array([[c_phi, -s_phi, 0, 0],\
                      [s_phi, c_phi, 0, 0] ,\
                      [0, 0, 1, 0],\
                      [0, 0, 0, 1]])
    return T_mat


# In[6]:


# get Local transformation matrices
T_1F = load_TF(phi_1to0)		
T_21 = load_T(a1, alphal, th_2t01, S_2t01)
T_32 = load_T(a2, alpha2, th_3t02, S_3t02)
T_43 = load_T(a3, alpha3, th_4t03, S_4t03)
T_54 = load_T(a4, alpha4, th_5t04, S_5t04)
T_65 = load_T(a5, alpha5, th_6t05, S_6t05)


# In[7]:


# get fixed transformation matrices
T_2F = np.matmul(T_1F, T_21)
T_3F = np.matmul(T_2F, T_32)
T_4F = np.matmul(T_3F, T_43)
T_5F = np.matmul(T_4F, T_54)
T_6F = np.matmul(T_5F, T_65)


# In[8]:


# get velocity state parameter vO_6to0
P_tool_inF = np.matmul(T_6F, np.append(P_tool_in6, 1.0))
print (f'P_tool_inF = {P_tool_inF}')
vO_6to0 = V_tool_inF - np.cross(omega_6to0, P_tool_inF[0:3])
print (f'desired velocity state = {omega_6to0}, {vO_6to0} ')


# In[9]:


# get line coordinates
S1 = T_1F[0:3,2]
SOL1 = np.cross(T_1F[0:3,3], T_1F[0:3,2])
print(f'S1;SOL1 = {S1} ; {SOL1}')
S2 = T_2F[0:3,2]
SOL2 = np.cross(T_2F[0:3,3], T_2F[0:3,2])
print(f'S2;SOL2 = {S2} ; {SOL2}')
S3 = T_3F[0:3,2]
SOL3 = np.cross(T_3F[0:3,3], T_3F[0:3,2])
print(f'S3;SOL3 = {S3} ; {SOL3}')
S4 = T_4F[0:3,2]
SOL4 = np.cross(T_4F[0:3,3], T_4F[0:3,2])
print(f'S4;SOL4 = {S4} ; {SOL4}')
S5 = T_5F[0:3,2]
SOL5 = np.cross(T_5F[0:3,3], T_5F[0:3,2])
print(f'S5;SOL5 = {S5} ; {SOL5}')
S6 = T_6F[0:3,2]
SOL6 = np.cross(T_6F[0:3,3], T_6F[0:3,2])
print(f'S6;SOL6 = {S6} ; {SOL6}')


# In[10]:


# 3 form up J matrix
Jmat = np.zeros((6,6))
Jmat[0:3,0] = S1
Jmat[3:6,0] = SOL1
Jmat[0:3,1] = S2
Jmat[3:6,1] = SOL2
Jmat[0:3,2] = S3
Jmat[3:6,2] = SOL3
Jmat[0:3,3] = S4
Jmat[3:6,3] = SOL4
Jmat[0:3,4] = S5
Jmat[3:6,4] = SOL5
Jmat[0:3,5] = S6
Jmat[3:6,5] = SOL6
np.round(Jmat,3)


# In[11]:


Jmat_inv = np.linalg.inv(Jmat)


# In[12]:


omega_values = np.matmul(Jmat_inv, np.append(omega_6to0, vO_6to0))
omega_values # rad/sec


# In[13]:


########################################################################


# In[14]:


# Start of acceleration problem
# desired acceleration state
alpha_6to0 = np.array([2.0, 0.5, 0]) # rad/sec^2
aO_6to0 = np.array([20, 0, -10]) # in/sec^2


# ![250413_eml6282_hw8A.jpg](attachment:ba3fe912-bceb-4d19-8180-c92ea183fb8e.jpg)

# ![250413_eml6282_hw8B.jpg](attachment:a103ce50-b657-4311-8d0a-8ea015ffb3ec.jpg)

# In[15]:


omega_1to0 = omega_values[0] * S1
vO_1to0    = omega_values[0] * SOL1
print(f'omega_1to0 ; vO_1to0 = {omega_1to0} ; {vO_1to0}')
omega_2to1 = omega_values[1] * S2
vO_2to1    = omega_values[1] * SOL2
print(f'omega_2to1 ; vO_2to1 = {omega_2to1} ; {vO_2to1}')
omega_3to2 = omega_values[2] * S3
vO_3to2    = omega_values[2] * SOL3
print(f'omega_3to2 ; vO_3to2 = {omega_3to2} ; {vO_3to2}')
omega_4to3 = omega_values[3] * S4
vO_4to3    = omega_values[3] * SOL4
print(f'omega_4to3 ; vO_4to3 = {omega_4to3} ; {vO_4to3}')
omega_5to4 = omega_values[4] * S5
vO_5to4    = omega_values[4] * SOL5
print(f'omega_5to4 ; vO_5to4 = {omega_5to4} ; {vO_5to4}')
omega_6to5 = omega_values[5] * S6
vO_6to5    = omega_values[5] * SOL6
print(f'omega_6to5 ; vO_6to5 = {omega_6to5} ; {vO_6to5}')


# In[16]:


alpha_6to0_R = alpha_6to0 \
    - np.cross(omega_1to0, omega_2to1 + omega_3to2 + omega_4to3 + omega_5to4 + omega_6to5)\
    - np.cross(omega_2to1, omega_3to2 + omega_4to3 + omega_5to4 + omega_6to5)\
    - np.cross(omega_3to2, omega_4to3 + omega_5to4 + omega_6to5)\
    - np.cross(omega_4to3, omega_5to4 + omega_6to5)\
    - np.cross(omega_5to4, omega_6to5)
print(f'alpha_6to0_R = {alpha_6to0_R} rad/sec^2')


# In[17]:


aO_6to0_R = aO_6to0 \
    - np.cross(omega_1to0, vO_1to0) - np.cross(omega_2to1, vO_2to1)\
    - np.cross(omega_3to2, vO_3to2) - np.cross(omega_4to3, vO_4to3)\
    - np.cross(omega_5to4, vO_5to4) - np.cross(omega_6to5, vO_6to5)\
    - np.cross(2*omega_1to0, vO_2to1 + vO_3to2 + vO_4to3 + vO_5to4 + vO_6to5)\
    - np.cross(2*omega_2to1, vO_3to2 + vO_4to3 + vO_5to4 + vO_6to5)\
    - np.cross(2*omega_3to2, vO_4to3 + vO_5to4 + vO_6to5)\
    - np.cross(2*omega_4to3, vO_5to4 + vO_6to5)\
    - np.cross(2*omega_5to4, vO_6to5)
print(f'aO_6to0_R = {aO_6to0_R} in/sec^2')


# In[18]:


alpha_values = np.matmul(Jmat_inv, np.append(alpha_6to0_R, aO_6to0_R))
alpha_values # rad/sec^2


# In[19]:


######################################


# In[20]:


print(f'omega_1to0 = {omega_values[0]:.4f} rad/sec \t= {omega_values[0]*R2D:.4f} deg/sec')
print(f'omega_2to1 = {omega_values[1]:.4f} rad/sec \t= {omega_values[1]*R2D:.4f} deg/sec')
print(f'omega_3to2 = {omega_values[2]:.4f} rad/sec \t= {omega_values[2]*R2D:.4f} deg/sec')
print(f'omega_4to3 = {omega_values[3]:.4f} rad/sec \t= {omega_values[3]*R2D:.4f} deg/sec')
print(f'omega_5to4 = {omega_values[4]:.4f} rad/sec \t= {omega_values[4]*R2D:.4f} deg/sec')
print(f'omega_6to5 = {omega_values[5]:.4f} rad/sec \t= {omega_values[5]*R2D:.4f} deg/sec')


# In[21]:


print(f'alpha_1to0 = {alpha_values[0]:.4f} rad/sec^2 \t= {alpha_values[0]*R2D:.4f} deg/sec^2')
print(f'alpha_2to1 = {alpha_values[1]:.4f} rad/sec^2 \t= {alpha_values[1]*R2D:.4f} deg/sec^2')
print(f'alpha_3to2 = {alpha_values[2]:.4f} rad/sec^2 \t= {alpha_values[2]*R2D:.4f} deg/sec^2')
print(f'alpha_4to3 = {alpha_values[3]:.4f} rad/sec^2 \t= {alpha_values[3]*R2D:.4f} deg/sec^2')
print(f'alpha_5to4 = {alpha_values[4]:.4f} rad/sec^2 \t= {alpha_values[4]*R2D:.4f} deg/sec^2')
print(f'alpha_6to5 = {alpha_values[5]:.4f} rad/sec^2 \t= {alpha_values[5]*R2D:.4f} deg/sec^2')


# In[ ]:



