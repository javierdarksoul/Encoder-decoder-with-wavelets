from scipy import signal
from scipy import misc
import collections
import heapq
import operator
import pickle
import json

import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap

#funcion que define la umbralizacion, se compara cada elemento de una matriz con una constante a, todos los valores menores a "a" son convertidos en 0
def umbral(a,lista):
    ind=list(lista.shape)
    for x in range(0,int(ind[0])):
        for y in range(0,int(ind[1])):
            if abs(lista[x][y][0])<a:
                lista[x][y][0]=0
            if abs(lista[x][y][1])<a:
                lista[x][y][1]=0
            if abs(lista[x][y][2])<a:
                lista[x][y][2]=0
    return lista
    
#Funcion de escalamiento, ocupada en la etapa de cuantificacion, produce valores con un rango de 2*sc
def escala(lista):
    sc=15
    ind=list(lista.shape)
    for c in range(0,3):
        for x in range(0,int(ind[0])):
            for y in range(0,int(ind[1])):
                lista[x][y][c]=int(lista[x][y][c]*sc)+(sc)

    return lista

#escala inversa, permite pasar de una transformacion escalada a una no escalada
def invescala(lista):
    sc=15
    ind=list(lista.shape)
    for c in range(0,3):
        for x in range(0,int(ind[0])):
            for y in range(0,int(ind[1])):
                lista[x][y][c]=(lista[x][y][c]-sc)/sc
    return lista
#funcion que realiza la operacion replica por filas a una matriz, esto significa repetir una vez cada elemento por fila [1.2] -> [1,1,2,2]
def inf(lista):
    ind=list(lista.shape)
    array=np.zeros((int(ind[0]),int(ind[1]*2)))
    for x in range(0,int(ind[0])):
        for y in range(0,int(ind[1])):
            array[x,2*y]=lista[x,y]
            array[x,(2*y)+1]=lista[x,y]
    return array
#funcion que realiza la operacion replica por columnas a una matriz.
def inc(lista):
    ind=list(lista.shape)
    array=np.zeros((int(ind[0])*2,int(ind[1])))
    for x in range(0,int(ind[0])):
        for y in range(0,int(ind[1])):
            array[2*x,y]=lista[x,y]
            array[(2*x)+1,y]=lista[x,y]
    return array
    
#funcion que calcula la transformada inversa de una matriz transformada
def reverse(lista):

    ind=list(lista.shape)
    rl=inc(lista[0:int(ind[0]/2),0:int(ind[1]/2),0])
    r1=inf(rl)

    r2=inc(lista[int(ind[0]/2):int(ind[0]),0:int(ind[1]/2),0])
    r2=inf(r2)

    r3=inc(lista[0:int(ind[0]/2),int(ind[1]/2):int(ind[1]),0])
    r3=inf(r3)

    r4=inc(lista[int(ind[0]/2):int(ind[0]),int(ind[1]/2):int(ind[1]),0])
    r4=inf(r4)

    gl=inc(lista[0:int(ind[0]/2),0:int(ind[1]/2),1])
    g1=inf(gl)

    g2=inc(lista[int(ind[0]/2):int(ind[0]),0:int(ind[1]/2),1])
    g2=inf(g2)

    g3=inc(lista[0:int(ind[0]/2),int(ind[1]/2):int(ind[1]),1])
    g3=inf(g3)

    g4=inc(lista[int(ind[0]/2):int(ind[0]),int(ind[1]/2):int(ind[1]),1])
    g4=inf(g4)

    b1=inc(lista[0:int(ind[0]/2),0:int(ind[1]/2),2])
    b1=inf(b1)

    b2=inc(lista[int(ind[0]/2):int(ind[0]),0:int(ind[1]/2),2])
    b2=inf(b2)

    b3=inc(lista[0:int(ind[0]/2),int(ind[1]/2):int(ind[1]),2])
    b3=inf(b3)

    b4=inc(lista[int(ind[0]/2):int(ind[0]),int(ind[1]/2):int(ind[1]),2])
    b4=inf(b4)

    for x in range(0,int(ind[0])):
        for y in range(0,int(ind[1])):
            if y%2!=0:
                r3[x,y]=r3[x,y]*-1
                g3[x,y]=g3[x,y]*-1
                b3[x,y]=b3[x,y]*-1
            if x%2!=0:
                r2[x,y]=r2[x,y]*-1
                g2[x,y]=g2[x,y]*-1
                b2[x,y]=b2[x,y]*-1
                if y%2==0:
                     r4[x,y]=r4[x,y]*-1
                     g4[x,y]=g4[x,y]*-1
                     b4[x,y]=b4[x,y]*-1
                
            if x%2==0:
                if y%2!=0:
                    r4[x,y]=r4[x,y]*-1
                    g4[x,y]=g4[x,y]*-1
                    b4[x,y]=b4[x,y]*-1


    i1=r1+r2+r3+r4
    i2=g1+g2+g3+g4
    i3=b1+b2+b3+b4
    kk=np.zeros((int(ind[0]),int(ind[1]),3))
    kk[:,:,0]=i1
    kk[:,:,1]=i2
    kk[:,:,2]=i3
    return kk

#funcion que aplica filtro pasa bajas por filas
def lowf(lista):
    ind=list(lista.shape)
    array=np.zeros((int(ind[0]),int(ind[1]/2)))
    for i in range(0,int(ind[0])):
        for j in range(0,int(ind[1]/2)):
            l=lista[i][2*j]+lista[i][(2*j)+1]
            l=1.0*l/2
            array[i][j]=l
    return array
    
#funcion que apica un filtro pasa altas por filas
def highf(lista):
    array=lowf(lista)
    ind=list(array.shape)
    array2=np.zeros((int(ind[0]),int(ind[1])))
    for x in range(0,int(ind[0])):
        for y in range(0,int(ind[1])):
            array2[x,y]=lista[x,2*y]-array[x,y]
    return array2

#funcion que aplica un filtro pasa altas por columnas a una matriz
def highc(lista):
    
    array=lowc(lista)
    ind=list(array.shape)
    array2=np.zeros((int(ind[0]),int(ind[1])))
    for x in range(0,int(ind[0])):
        for y in range(0,int(ind[1])):
            array2[x,y]=lista[2*x,y]-array[x,y]
    return array2        

#funcion que aplica un filtro pasa bajas por columnas a una matriz
def lowc(lista):
    ind=list(lista.shape)
    array=np.zeros((int(ind[0]/2),int(ind[1])))
    for i in range(0,int(ind[0]/2)):
        for j in range(0,ind[1]):
            l=(lista[2*i][j]+lista[(2*i)+1][j])
            l=1.0*l/2
            array[i][j]=l
    return array

#funcion que calcula un filtro pasa bajas por filas , luego, a esa matriz filtrada, se le aplica un filtro pasa bajas por columnas
def LL(d):
    de=list(d.shape)
    g=np.zeros((de[0],de[1]))
    h=np.zeros((de[0],de[1]))
    i=np.zeros((de[0],de[1]))

    for x in range(0,de[0]):
        for y in range(0,de[1]):
            g[x][y]= d[x][y][0]
            h[x][y]= d[x][y][1]
            i[x][y]= d[x][y][2]
        
    arr=lowf(g)
    arr=lowc(arr)

    arr2=lowf(h)
    arr2=lowc(arr2)

    arr3=lowf(i)
    arr3=lowc(arr3)

    defe=list(arr.shape)
    k=np.zeros((defe[0],defe[1],3))
    for x in range(0,defe[0]):
        for y in range(0,defe[1]):
            k[x][y][0]=arr[x][y]
            k[x][y][1]=arr2[x][y]
            k[x][y][2]=arr3[x][y]
    return k

#funcion que calcula un filtro pasa bajas por filas , luego, a esa matriz filtrada, se le aplica un filtro pasa altas por columnas
def LH(d):
    de=list(d.shape)
    g=np.zeros((de[0],de[1]))
    h=np.zeros((de[0],de[1]))
    i=np.zeros((de[0],de[1]))

    for x in range(0,de[0]):
        for y in range(0,de[1]):
            g[x][y]= d[x][y][0]
            h[x][y]= d[x][y][1]
            i[x][y]= d[x][y][2]
        
    arr=lowf(g)
    arr=highc(arr)

    arr2=lowf(h)
    arr2=highc(arr2)

    arr3=lowf(i)
    arr3=highc(arr3)

    defe=list(arr.shape)
    k=np.zeros((defe[0],defe[1],3))
    for x in range(0,defe[0]):
        for y in range(0,defe[1]):
            k[x][y][0]=arr[x][y]
            k[x][y][1]=arr2[x][y]
            k[x][y][2]=arr3[x][y]
    return k
    
#funcion que calcula un filtro pasa altas por filas , luego, a esa matriz filtrada, se le aplica un filtro pasa bajas por columnas
def HL(d):
    de=list(d.shape)
    g=np.zeros((de[0],de[1]))
    h=np.zeros((de[0],de[1]))
    i=np.zeros((de[0],de[1]))

    for x in range(0,de[0]):
        for y in range(0,de[1]):
            g[x][y]= d[x][y][0]
            h[x][y]= d[x][y][1]
            i[x][y]= d[x][y][2]
        
    arr=highf(g)
    arr=lowc(arr)

    arr2=highf(h)
    arr2=lowc(arr2)

    arr3=highf(i)
    arr3=lowc(arr3)

    defe=list(arr.shape)
    k=np.zeros((defe[0],defe[1],3))
    for x in range(0,defe[0]):
        for y in range(0,defe[1]):
            k[x][y][0]=arr[x][y]
            k[x][y][1]=arr2[x][y]
            k[x][y][2]=arr3[x][y]
    return k
#funcion que calcula un filtro pasa altas por filas , luego, a esa matriz filtrada, se le aplica un filtro pasa altas por columnas
def HH(d):
    de=list(d.shape)
    g=np.zeros((de[0],de[1]))
    h=np.zeros((de[0],de[1]))
    i=np.zeros((de[0],de[1]))

    for x in range(0,de[0]):
        for y in range(0,de[1]):
            g[x][y]= d[x][y][0]
            h[x][y]= d[x][y][1]
            i[x][y]= d[x][y][2]
        
    arr=highf(g)
    arr=highc(arr)

    arr2=highf(h)
    arr2=highc(arr2)

    arr3=highf(i)
    arr3=highc(arr3)

    defe=list(arr.shape)
    k=np.zeros((defe[0],defe[1],3))
    for x in range(0,defe[0]):
        for y in range(0,defe[1]):
            k[x][y][0]=arr[x][y]
            k[x][y][1]=arr2[x][y]
            k[x][y][2]=arr3[x][y]
    return k

#Calcula la wavelet a una matriz de entrada y aplica umbral
def wavelet(fr,ind):
    fr2=fr[0:int(ind[0]),0:int(ind[1]),:]
    k=LL(fr2)
    k1=LH(fr2)
    k1=umbral(0.01,k1)
    k2=HL(fr2)
    k2=umbral(0.01,k2)
    k3=HH(fr2)
    k3=umbral(0.01,k3)
    
    fr[0:int(ind[0]/2),0:int(ind[1]/2),:]=k[:,:,:]
    fr[int(ind[0]/2):int(ind[0]),0:int(ind[1]/2),:]=k1[:,:,:]
    fr[0:int(ind[0]/2),int(ind[1]/2):int(ind[1]),:]=k2[:,:,:]
    fr[int(ind[0]/2):int(ind[0]),int(ind[1]/2):int(ind[1]),:]=k3[:,:,:]
    return fr
    
#Calcula la wavelet de nivel n                
def wavelets(d,n):
    ind=list(d.shape)
    ind=[ind[0],ind[1],3]
    for x in range(0,n):
        d=wavelet(d,ind)
        ind=[ind[0]/2,ind[1]/2,3]
    return d

#calcula la transformada inversa para una wavelet de nivel n
def inverseM(k,n):
    ind=list(k.shape)
    for f in range(1,n+1):
        g=int(ind[0])
        g=g/2**(n-f)
        g=int(g)
        h=int(ind[1])/2**(n-f)
        h=int(h)
        k2=reverse(k[0:g,0:h,:])
        k[0:g,0:h,:]=k2
    return k

#Transforma un string proveniente a binario segun la tabla de codificacion de Huffman, luego retorna el entero de dicho binario.
def compress(dic,content):
    res = ""
    for ch in content:
        code = dic[ch]
        res = res + code
    res = '1' + res + dic['0']
    res = res + (len(res) % 8 * "0")
    return int(res,2)

#Guarda en un archivo los datos transformados con Huffman.
def store(k,dic,outfile):
    ind=list(k.shape)
    res=[]
    outf = open(outfile,'wb')
    for c in range(0,3):
        for y in range(0,int(ind[1])):
            res=[]
            for x in range(0,int(ind[0])):
                res.append(str(int(k[x,y,c])))
            com=compress(dic,res)
            pickle.dump(com,outf)
    outf.close()

#Funcion que recibe la cadena de binarios para ser transformada de vuelta en la matriz (imagen)
def decodif(dic, bitstr):
    res = []
    length = bitstr.bit_length() - 1
    if bitstr >> length != 1:
        raise Error("Corrupt file!")
    done = False
    while length > 0 and not done:
        shift = length - 1
        while True:
            num = bitstr >> shift
            bitnum = bin(num)[3:]
            if bitnum not in dic.values():
                shift -= 1
                continue
            char = list(dic.keys())[list(dic.values()).index(bitnum)]
            if char == '0':
                done = True
                break
            res.append(char)
            bitstr = bitstr - ((num - 1) << shift)
            length = shift
    return res
    
#Lee el archivo .huijse y llama a la funcion decodif para decodificar
def decode(ind,dic,file):
    f = open(file,'rb')
    objects = []
    while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    f.close()
    h = np.zeros((int(ind[0]),int(ind[1]),3))

    for c in range(0,3):
        for y in range(0,int(ind[1])):
            jue = decodif(dic,objects[(int(ind[1])*(c))+y])
            cont = 0
            for x in range(0,int(ind[0])):
                h[x,y,c]=jue[cont]
                cont=cont+1
    return h


#dic = {'50': '1','49': '0000', '48': '00010', '73': '000110000', '81': '000110001', '68': '00011001', '60': '0001101', '57': '000111', '52': '0010', '76': '001100000', '83': '0011000010', '41': '0011000011', '44': '00110001', '67': '00110010', '77': '001100110', '72': '001100111', '47': '001101', '55': '00111', '53': '0100', '36': '0101000000000', '100': '01010000000010', '33': '0101000000001100', '1': '010100000000110100', '2': '010100000000110101', '3': '010100000000110110', '4': '010100000000110111', '5': '010100000000111000', '6': '010100000000111001', '7': '010100000000111010', '8': '010100000000111011', '9': '010100000000111100', '10': '010100000000111101', '11': '010100000000111110', '12': '010100000000111111', '38': '010100000001', '98': '01010000001', '93': '01010000010', '13': '010100000110000000', '14': '010100000110000001', '15': '010100000110000010', '16': '010100000110000011', '17': '010100000110000100', '18': '010100000110000101', '19': '010100000110000110', '20': '010100000110000111', '21': '010100000110001000', '22': '010100000110001001', '23': '010100000110001010', '24': '010100000110001011', '25': '010100000110001100', '26': '010100000110001101', '27': '010100000110001110', '28': '010100000110001111', '34': '0101000001100100', '29': '010100000110010100', '30': '010100000110010101', '32': '010100000110010110', '31': '0101000001100101110', '0': '0101000001100101111', '35': '010100000110011', '37': '0101000001101', '96': '010100000111', '78': '010100001', '66': '01010001', '46': '0101001', '65': '01010100', '80': '010101010', '43': '010101011', '59': '0101011', '56': '010110', '79': '010111000', '91': '0101110010', '40': '01011100110', '97': '010111001110', '95': '010111001111', '99': '0101110100', '82': '0101110101', '89': '0101110110', '92': '01011101110', '85': '01011101111', '64': '01011110', '62': '01011111', '51': '0110', '71': '011100000', '86': '01110000100', '87': '01110000101', '42': '0111000011', '63': '01110001', '45': '01110010', '70': '011100110', '74': '0111001110', '84': '01110011110', '39': '011100111110', '94': '011100111111', '58': '0111010', '61': '01110110', '88': '01110111000', '90': '01110111001', '75': '0111011101', '69': '011101111', '54': '01111'}
#dic={'25': '1', '28': '00000', '43': '0000100', '50': '0000101', '41': '000011', '33': '000100', '31': '000101', '23': '00011', '30': '001000', '32': '001001', '40': '001010', '49': '001011', '21': '0011000', '47': '00110010', '44': '00110011', '34': '001101', '39': '001110', '29': '001111', '38': '010000', '36': '010001', '37': '010010', '35': '010011', '27': '01010', '48': '01011000', '45': '01011001', '42': '0101101', '46': '01011100', '16': '0101110100000', '0': '01011101000010000', '1': '01011101000010001', '2': '01011101000010010', '3': '01011101000010011', '4': '01011101000010100', '5': '01011101000010101', '6': '01011101000010110', '7': '01011101000010111', '8': '01011101000011000', '9': '01011101000011001', '10': '01011101000011010', '11': '01011101000011011', '12': '01011101000011100', '13': '01011101000011101', '14': '01011101000011110', '15': '01011101000011111', '17': '010111010001', '18': '01011101001', '19': '0101110101', '20': '010111011', '22': '0101111', '24': '0110', '26': '0111'}
dic={'15': '1','13': '00000', '28': '000010', '27': '000011', '17': '0001', '14': '00100', '25': '00101', '29': '00110', '26': '001110', '6': '0011110000000', '7': '0011110000001', '9': '001111000001', '8': '0011110000100', '0': '00111100001010', '1': '00111100001011', '2': '00111100001100', '3': '00111100001101', '4': '00111100001110', '5': '00111100001111', '10': '0011110001', '11': '001111001', '12': '00111101', '30': '0011111', '18': '01000', '19': '01001', '24': '01010', '16': '01011', '20': '01100', '23': '01101', '21': '01110', '22': '01111'}

#Llenar arreglo files con los archivos a procesar (no poner extencion)
files=[]
for y in range(0,94):
    files.append("frame0"+str(628+y))
    
for x in range(0,len(files)):
    d=plt.imread("frames/"+files[x]+".png")
    d=d[:,:,0:3]
    file="intermedio/"+"hola"

    print("Aplicando Transformada... ",end="")
    k=wavelets(d,2)
    print("Listo!")

    print("     Cuantizando... ",end="")
    k=escala(k)
    print("Listo!")

    print("          Codificando... ",end="")
    store(k,dic,"intermedio/"+files[x]+".huijse")
    print("Listo!")

    print("               Archivo "+ files[x]+".huijse creado")

    print("          Decodificando... ",end="")
    h=decode(list(k.shape),dic,"intermedio/"+files[x]+".huijse")
    print("Listo!")

    print("     Decuantizando... ",end="")
    h=invescala(h)
    print("Listo!")

    print("Aplicando Inversa de la Transfomada... ",end="")
    h=inverseM(h,2)
    print("Listo!")

    plt.imsave(fname="output/"+files[x]+".png",arr=h)
