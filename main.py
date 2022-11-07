import numpy as np
from matplotlib import pyplot as plt
# k >= 1
def Int1(n,k): #L(phi_i,phi_j)

    h=float(1)/(n-1)
    return 8*k/(3*h)+16/15*h

def Int2(n,k): #L(psi_i,psi_i) OK
    h=float(1)/(n-1)
    return 16*n*k/3+8*h/15

def Int3(n,k): ##L(phi_j,psi_i)
    h=float(1)/(n-1)
    return 4*n*k/3+7/(15*n)+2/3

def Int3_1(n,k): #L(psi_j,phi_i)
    h=float(1)/(n-1)
    return 4*n*k/3+7/(15*n)-2/3

def Int4(n,k): #L(psi_j,phi_i)
    h=float(1)/(n-1)
    return 16*n*k/3+7/15/n-2/3

def Int5(n,k): #???
    h=float(1)/(n-1)
    return -2*k*n/3+11/30/n + 5/6 #the last add is mine

def b2(i:int,n:int,f)->float:
    h=float(1)/(n-1)
    return 4/3*h*f(h*i)

def b1(i:int,n:int,f)->float:
    h=float(1)/(n-1)
    return 2/3*h*f(h*i)

def function(x:float, k:float)->float:
    var=(np.pi* np.sin(2*np.pi*x)-1/2* (1 + 4*k*(np.pi**2))*np.cos(2*np.pi*x) + 1/2)
    return var

def b2(i,n,function,k):
    h=float(1)/(n-1)
    return 2/3*h*function(h*i,k)

def b1(i,n,function,k):
    h=float(1)/(n-1)
    return 4/3*h*function(h*i,k)

def filling(n: int, a: np.ndarray,b:np.ndarray): #coefficients a_ij, b_i
    #h = float(1) / (n-1)
    h = float(1) / (n)
    k = 5 / 2
    #print(k)
    #boundary condition1
    a[0][0] = -2+1/h+h/2
    a[0][1] = 1
    a[0][2] = -1/h
    b[0]= h/2* function(0,k)
    #boundary condition2
    a[2*n][2*n]=1
    b[2*n]=0

    for i in range(1, n):
        for j in range(1, 2*n):
            if (i%2==1):
                b[i]=b2(i,n,function,k)
                if(i==j):
                    a[i][j]=Int2(n,k)
                elif (i==j-1):
                    a[i][j]=Int3_1(n,k)
                elif(i==j+1):
                    a[i][j]=Int3(n,k)
            else:
                b[i]=b1(i,n,function,k)
                if(i==j):
                    a[i][j]=Int1(n,k)
                elif (i == j + 2):
                    a[i][j] = Int5(n, k)
                elif (i == j -2):
                    a[i][j] = Int5(n, k)
                elif (i == j - 1):
                    a[i][j] = Int3_1(n, k)
                elif (i == j + 1):
                    a[i][j] = Int3(n, k)
    a[1][0]=Int3(n,k)
    k=3
    for i in range(n, 2*n):
        for j in range(1,2*n):
            if (i % 2 == 1):
                b[i] = b2(i, n, function,k)
                if (i == j):
                    a[i][j] = Int2(n, k)
                elif (i == j - 1):
                    a[i][j] = Int3_1(n, k)
                elif (i == j + 1):
                    a[i][j] = Int3(n, k)
            else:
                b[i] = b1(i, n, function,k)
                if (i == j):
                    a[i][j] = Int1(n, k)
                elif (i == j + 2):
                    a[i][j] = Int5(n, k)
                elif (i == j - 2):
                    a[i][j] = Int5(n, k)
                elif (i == j - 1):
                    a[i][j] = Int3_1(n, k)
                elif (i == j + 1):
                    a[i][j] = Int3(n, k)
    return a,b

def f(x, c):
    h = float(1) / ((len(c) - 1)/2)
    i = int(np.floor(x / h))
    j = int(np.ceil(x / h))
    if (i != j):
        return (-4*((1/h)**2)*((x-(i+1/2)/(1/h))**2)+1)*c[2*i+1]+(-(((1/h)**2)*((x-(i)/(1/h))**2))+1)*c[2*i]+(-(((1/h)**2)*((x-(i+1)/(1/h))**2))+1)*c[2*(i+1)]
    else:
        return c[2*i]

def result_c(n):
    size = 2 * n + 1
    #h = float(1) / (n)

    a = np.zeros((size, size), dtype=float)
    b = np.zeros(size,dtype=float)

    filling(n,a,b)
    #print(a)
    c = np.linalg.solve(a, b)
    return c


x = np.linspace(0, 1, 1000)
y = np.zeros(len(x))
for i in range(0, len(x)):
    y = np.sin(np.pi * x) ** 2

y1 = np.zeros(len(x))
fig, ax = plt.subplots(3, 2, figsize=[10, 11])
plt.suptitle('sin(pi * x)**2', y=1.01, fontsize=16)

c = result_c(5)
for i in range(0, len(x)):
    y1[i] = f(x[i], c)
ax[0][0].plot(x, y1)
ax[0][0].plot(x, y)
ax[0][0].text(0.1, 0.5, 'n = 5', fontsize=12, transform=ax[0][0].transAxes)

c = result_c(10)
for i in range(0, len(x)):
    y1[i] = f(x[i], c)
ax[0][1].plot(x, y1)
ax[0][1].plot(x, y)
ax[0][1].text(0.1, 0.5, 'n = 10', fontsize=12, transform=ax[0][1].transAxes)

c = result_c(20)
for i in range(0, len(x)):
    y1[i] = f(x[i], c)
ax[1][0].plot(x, y1)
ax[1][0].plot(x, y)
ax[1][0].text(0.1, 0.5, 'n = 20', fontsize=12, transform=ax[1][0].transAxes)

c = result_c(50)
for i in range(0, len(x)):
    y1[i] = f(x[i], c)
ax[1][1].plot(x, y1)
ax[1][1].plot(x, y)
ax[1][1].text(0.1, 0.5, 'n = 50', fontsize=12, transform=ax[1][1].transAxes)

# c = result_c(100)
# for i in range(0, len(x)):
#     y1[i] = f(x[i], c)
# ax[2][0].plot(x, y1)
# ax[2][0].plot(x, y)
# ax[2][0].text(0.1, 0.5, 'n = 100', fontsize=12, transform=ax[2][0].transAxes)
#
# c = result_c(200)
# for i in range(0, len(x)):
#     y1[i] = f(x[i], c)
# ax[2][1].plot(x, y1)
# ax[2][1].plot(x, y)
# ax[2][1].text(0.1, 0.5, 'n = 200', fontsize=12, transform=ax[2][1].transAxes)
c = result_c(500)
for i in range(0, len(x)):
    y1[i] = f(x[i], c)
ax[2][1].plot(x, y1)
ax[2][1].plot(x, y)
ax[2][1].text(0.1, 0.5, 'n = 500', fontsize=12, transform=ax[2][1].transAxes)

c = result_c(1000)
for i in range(0, len(x)):
    y1[i] = f(x[i], c)
ax[2][1].plot(x, y1)
ax[2][1].plot(x, y)
ax[2][1].text(0.1, 0.5, 'n = 1000', fontsize=12, transform=ax[2][1].transAxes)

fig.tight_layout()
plt.show()
# plt.savefig('Task1.png')
