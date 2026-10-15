import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy import sparse

def nodeij(i,im):
    inode = i%(im-1)
    jnode = i//(im-1)
    return inode,jnode
def nodeij2(i,im):
    inode = i%(im+1)
    jnode = i//(im+1)
    return inode,jnode

Re=2000
Pr=7
im=81
jm=81
H=1
dx=H/im
dy=H/jm
beta=dx/dy
N=(im-1)*(jm-1)
N2=(im+1)*(jm+1)

NZ=4*3+2*(im-3+jm-3)*4+(im-3)*(im-3)*5
NZ2=4*2+2*(im-1)+2*2*(jm-1)+(im-1)*(jm-1)*5
dataomega=np.zeros(NZ) ; rowomega=np.zeros(NZ,int) ; colomega=rowomega.copy() ; bomega=np.zeros(N)
datasai=dataomega.copy() ; rowsai=rowomega.copy() ; colsai=rowomega.copy() ; bsai=bomega.copy()
datatemp=np.zeros(NZ2) ; rowtemp=np.zeros(NZ2,int) ; coltemp=rowtemp.copy() ; btemp=np.zeros(N2)
uold=np.zeros(N) ; unew=uold.copy()
vold=uold.copy() ; vnew=uold.copy()
saiold=uold.copy() ; sainew=uold.copy()
omegaold=uold.copy() ; omeganew=uold.copy()

u=np.zeros(N)
v=u.copy()
sai=u.copy()
omega=u.copy()
temp=np.zeros(N2)

saiwall=0

komega=0.1
ksai=0.1
er_array=np.zeros([4,1])
erj=np.zeros([4,1])

j=0
er=1
while er>1e-8:
    j+=1
    # calculating vorticity field
    inz=-1
    for i in range(N):
        inode,jnode = nodeij(i,im)
        inz+=1
        dataomega[inz]=max(uold[i],0)+max(-uold[i],0)+beta*max(vold[i],0)+beta*max(-vold[i],0)+2/(Re*dx)+2*beta**2/(Re*dx)
        rowomega[inz]=i
        colomega[inz]=i
        bomega[i]=0
        if inode==0:
            inz+=1
            dataomega[inz]=-max(-uold[i],0)-1/(Re*dx)
            rowomega[inz]=i
            colomega[inz]=i+1
            bomega[i]+=2/(dx**2)*(max(uold[i],0)+1/(Re*dx))*(saiwall-saiold[i])
        elif inode==im-2:
            inz+=1
            dataomega[inz]=-(max(uold[i],0)+1/(Re*dx))
            rowomega[inz]=i
            colomega[inz]=i-1
            bomega[i]+=2/(dx**2)*(max(-uold[i],0)+1/(Re*dx))*(saiwall-saiold[i])
        else:
            inz+=1
            dataomega[inz]=-(max(-uold[i],0)+1/(Re*dx))
            rowomega[inz]=i
            colomega[inz]=i+1
            inz+=1
            dataomega[inz]=-(max(uold[i],0)+1/(Re*dx))
            rowomega[inz]=i
            colomega[inz]=i-1
            bomega[i]+=0    
        if jnode==0:
            inz+=1
            dataomega[inz]=-(beta*max(-vold[i],0)+beta**2/(Re*dx))
            rowomega[inz]=i
            colomega[inz]=i+(im-1)
            bomega[i]+=2/(dy**2)*(beta*max(vold[i],0)+beta**2/(Re*dx))*(saiwall-saiold[i])
        elif jnode==jm-2:
            inz+=1
            dataomega[inz]=-(beta*max(vold[i],0)+beta**2/(Re*dx))
            rowomega[inz]=i
            colomega[inz]=i-(im-1)
            bomega[i]+=2/(dy**2)*(beta*max(-vold[i],0)+beta**2/(Re*dx))*(saiwall-saiold[i]-1*dy)
        else:
            inz+=1
            dataomega[inz]=-(beta*max(-vold[i],0)+beta**2/(Re*dx))
            rowomega[inz]=i
            colomega[inz]=i+(im-1)
            inz+=1
            dataomega[inz]=-(beta*max(vold[i],0)+beta**2/(Re*dx))
            rowomega[inz]=i
            colomega[inz]=i-(im-1)
            bomega[i]+=0
    a2=sparse.csr_matrix((dataomega, (rowomega, colomega)), shape=(N, N))
    x=spla.gmres(a2,bomega,omegaold)
    omeganew=x[0]
    c1=x[1]
    er_omega=np.sqrt(np.sum((omeganew-omegaold)**2)/np.sum(omeganew**2))
    erj[0,0]=er_omega
    omeganew=omegaold+komega*(omeganew-omegaold)
    
    # calculating stream function
    inz=-1
    for i in range(N):
        inode,jnode = nodeij(i,im)
        inz+=1
        datasai[inz]=-2*(1+beta**2)
        rowsai[inz]=i
        colsai[inz]=i
        bsai[i]=-dx**2*omeganew[i]
        if inode==0:
            inz+=1
            datasai[inz]=1
            rowsai[inz]=i
            colsai[inz]=i+1
            bsai[i]+=-saiwall
        elif inode==im-2:
            inz+=1
            datasai[inz]=1
            rowsai[inz]=i
            colsai[inz]=i-1
            bsai[i]+=-saiwall
        else:
            inz+=1
            datasai[inz]=1
            rowsai[inz]=i
            colsai[inz]=i+1
            inz+=1
            datasai[inz]=1
            rowsai[inz]=i
            colsai[inz]=i-1
            bsai[i]+=0
        if jnode==0:
            inz+=1
            datasai[inz]=beta**2
            rowsai[inz]=i
            colsai[inz]=i+(im-1)
            bsai[i]+=-beta**2*saiwall
        elif jnode==jm-2:
            inz+=1
            datasai[inz]=beta**2
            rowsai[inz]=i
            colsai[inz]=i-(im-1)
            bsai[i]+=-beta**2*saiwall
        else:
            inz+=1
            datasai[inz]=beta**2
            rowsai[inz]=i
            colsai[inz]=i+(im-1)
            inz+=1
            datasai[inz]=beta**2
            rowsai[inz]=i
            colsai[inz]=i-(im-1)
            bsai[i]+=0
    a2=sparse.csr_matrix((datasai, (rowsai, colsai)), shape=(N, N))
    x=spla.gmres(a2,bsai,saiold)
    sainew=x[0]
    c2=x[1]
    er_sai=np.sqrt(np.sum((sainew-saiold)**2)/np.sum(sainew**2))
    erj[1,0]=er_sai
    sainew=saiold+ksai*(sainew-saiold)
        
    # calculating x & y velocities
    for i in range(N):
        inode,jnode = nodeij(i,im)
        if inode==0:
            vnew[i]=-(sainew[i+1]-saiwall)/(2*dx)
        elif inode==im-2:
            vnew[i]=-(saiwall-sainew[i-1])/(2*dx)
        else:
            vnew[i]=-(sainew[i+1]-sainew[i-1])/(2*dx)
        if jnode==0:
            unew[i]=(sainew[i+(im-1)]-saiwall)/(2*dy)
        elif jnode==jm-2:
            unew[i]=(saiwall-sainew[i-(im-1)])/(2*dy)
        else:
            unew[i]=(sainew[i+(im-1)]-sainew[i-(im-1)])/(2*dy)
    er_u=np.sqrt(np.sum((unew-uold)**2)/np.sum(unew**2))
    erj[2,0]=er_u
    er_v=np.sqrt(np.sum((vnew-vold)**2)/np.sum(vnew**2))
    erj[3,0]=er_v
    er_array=np.concatenate((er_array, erj), axis=1)
    er=max(erj)
#    if er<1e-2:
#        ksai=0.05
#        komega=0.05
#    elif er<1e-3:
#        ksai=0.01
#        komega=0.01
    print(j,er,c1,c2)
    # upgrading old variables for unknowns (k->k+1)
    omegaold=omeganew.copy()
    saiold=sainew.copy()
    uold=unew.copy()
    vold=vnew.copy()
# assigning most recent calculated variables (last k) to final solution
omega=omeganew
sai=sainew
u=unew
v=vnew

# calculating temperature field
inz=-1
for i in range(N2):
    inode,jnode = nodeij2(i,im)
    if inode==0:
        inz+=1
        datatemp[inz]=1
        rowtemp[inz]=i
        coltemp[inz]=i
        inz+=1
        datatemp[inz]=-1
        rowtemp[inz]=i
        coltemp[inz]=i+1
    elif inode==im:
        inz+=1
        datatemp[inz]=1
        rowtemp[inz]=i
        coltemp[inz]=i
        inz+=1
        datatemp[inz]=-1
        rowtemp[inz]=i
        coltemp[inz]=i-1
    else:
        if jnode==0:
            inz+=1
            datatemp[inz]=1
            rowtemp[inz]=i
            coltemp[inz]=i
        elif jnode==jm:
            inz+=1
            datatemp[inz]=1
            rowtemp[inz]=i
            coltemp[inz]=i
            btemp[i]=1
        else:
            ip=(jnode-1)*(im-1)+inode-1
            inz+=1
            datatemp[inz]=max(u[ip],0)+max(-u[ip],0)+beta*max(v[ip],0)+beta*max(-v[ip],0)+2/(Re*Pr*dx)+2*beta**2/(Re*Pr*dx)
            rowtemp[inz]=i
            coltemp[inz]=i
            inz+=1
            datatemp[inz]=-max(-u[ip],0)-1/(Re*Pr*dx)
            rowtemp[inz]=i
            coltemp[inz]=i+1
            inz+=1
            datatemp[inz]=-(max(u[ip],0)+1/(Re*Pr*dx))
            rowtemp[inz]=i
            coltemp[inz]=i-1
            inz+=1
            datatemp[inz]=-(beta*max(-v[ip],0)+beta**2/(Re*Pr*dx))
            rowtemp[inz]=i
            coltemp[inz]=i+(im+1)
            inz+=1
            datatemp[inz]=-(beta*max(v[ip],0)+beta**2/(Re*Pr*dx))
            rowtemp[inz]=i
            coltemp[inz]=i-(im+1)
a2=sparse.csr_matrix((datatemp, (rowtemp, coltemp)), shape=(N2, N2))
x=spla.gmres(a2,btemp)
temp=x[0]
c3=x[1]
print(c3)

x=np.linspace(0,1,im+1)
y=np.linspace(0,1,jm+1)
X,Y=np.meshgrid(x,y)

U=np.zeros([jm+1,im+1])
U[1:jm,1:im]=np.reshape(u,[jm-1,im-1])
U[jm,:]=1
plt.figure(num=0,figsize=(8,6))
cp=plt.contourf(X,Y,U,35,cmap='jet')
plt.colorbar()

V=np.zeros([jm+1,im+1])
V[1:jm,1:im]=np.reshape(v,[jm-1,im-1])
plt.figure(num=1,figsize=(8,6))
cp=plt.contourf(X,Y,V,35,cmap='jet')
plt.colorbar()

plt.figure(num=2,figsize=(6,6))
plt.quiver(X,Y,U,V)

plt.figure(num=3,figsize=(6,6))
plt.streamplot(X,Y,U,V,color=U**2+V**2,cmap='jet')

SAI=np.zeros([jm+1,im+1])+saiwall
SAI[1:jm,1:im]=np.reshape(sai,[jm-1,im-1])
plt.figure(num=4,figsize=(6,6))
levels=np.array([-0.1175,-0.115,-0.11,-0.09,-0.07,-0.05,-0.03,-0.01,-1e-4,-1e-5,-1e-7,-1e-10\
                 ,1e-8,1e-7,1e-6,1e-5,5e-5,1e-4,2.5e-4,5e-4,1e-3,1.5e-3,3e-3],float)
cp=plt.contour(X,Y,SAI,levels,cmap='jet')
plt.clabel(cp, inline=True, fontsize=10)

OMEGA=np.zeros([jm+1,im+1])
OMEGA[1:jm,1:im]=np.reshape(omega,[jm-1,im-1])
OMEGA[:,0]=2/(dx**2)*(saiwall-SAI[:,1])
OMEGA[:,im]=2/(dx**2)*(saiwall-SAI[:,im-1])
OMEGA[0,:]=2/(dy**2)*(saiwall-SAI[1,:])
OMEGA[jm,:]=2/(dy**2)*(saiwall-SAI[jm-1,:]-1*dy)
plt.figure(num=5,figsize=(6,6))
levels=np.array([-3,-2,-1,-0.5,0,0.5,1,2,3,4,5])
cp=plt.contour(X,Y,OMEGA,levels,cmap='jet')
plt.clabel(cp, inline=True, fontsize=10)

TEMP=np.reshape(temp,[jm+1,im+1])
plt.figure(num=6,figsize=(6,6))
#levels=np.array([-3,-2,-1,-0.5,0,0.5,1,2,3,4,5])
cp=plt.contour(X,Y,TEMP,20,cmap='jet')
plt.clabel(cp, inline=True, fontsize=10)

Nux=-(TEMP[jm,1:im]-TEMP[jm-1,1:im])/dy
Nuavg=sum(Nux)/(im-1)
print('Average Nusselt on top plane:',Nuavg)
plt.figure(7)
plt.plot(x[1:im],Nux)
plt.xlabel('x') ; plt.ylabel('Nux') ; plt.grid(True)

plt.figure(8)
plt.plot(er_array[:,1::].T)
plt.yscale('log') ; plt.grid(True)
plt.xlabel('k') ; plt.ylabel('relative rms')
plt.legend(('vorticity','stream function','x-velocity','y-velocity'))

plt.figure(9)
plt.plot(x,OMEGA[0,:], x,OMEGA[jm,:], y,OMEGA[:,0], y,OMEGA[:,im])
plt.grid(True)
plt.xlabel('x or y') ; plt.ylabel('vorticity')
plt.legend(('bottom','top','left','right'))

mx=int((im+1)/2)
my=int((jm+1)/2)

plt.figure(10)
plt.plot(x,U[my,:], x,V[my,:])
plt.grid(True)
plt.xlabel('x') ; plt.ylabel('velocity')
plt.legend(('u','v'))

plt.figure(11)
plt.plot(y,U[:,mx], y,V[:,mx])
plt.grid(True)
plt.xlabel('y') ; plt.ylabel('velocity')
plt.legend(('u','v'))

plt.figure(12)
plt.plot(x,TEMP[my,:])
plt.grid(True)
plt.xlabel('x') ; plt.ylabel('temperature')

plt.figure(13)
plt.plot(y,TEMP[:,mx])
plt.grid(True)
plt.xlabel('y') ; plt.ylabel('temperature')