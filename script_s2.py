
import numpy as np
import cmath
import scipy.weave as wv
import numpy.lib.scimath as smath
import matplotlib.pyplot as plt
import math
import sys
from PIL import Image
import time

#enter values of [f0,f] into this array to get a band structure for those parameters
frequency_values = [[0.4,0.2]]#,[0.6,0.2],[0.6,0.05],[0.3,0.5],[0.8,0.5] ]

#enter the desired graph resolution
dim = 100000000 #the number of steps on the x and y axes. dim^2 is the total number of points evaluated, so this grows quadratically with an increase in dim



#Defining fundamental constants
e = 1.60217662 * (10^-19) #electron charge
omega_l = 1 #laser frequency - set to 1 for simplicity of graphing omega/omega_l
c = 3 * (10^8) #speed of light
eps0 = 9.954187817 * (10^-12) #electric permittivity
me = 9.10938356 * (10^-31) #electron mass
nc = ((eps0*me*omega_l*omega_l)/(math.pow(e,2))) #critical density of plasma

n0,n1,n2,omega_pe_1,omega_pe_2,l,h1,h2 = 0,0,0,0,0,0,0,0



#setting up empty 2D arrays, of size (dim x dim), to hold whether K is complex or real at that point
#@profile
def calculate_constants(f0, f):
    t0 = time.time()
    global n0,n1,n2,omega_pe_1,omega_pe_2,l,h1,h2
    n0 = f0*nc #initial density of plasma, defined by setting f0
    n1= (1-f)*n0 #density of PPC layer 1
    n2=(1+f)*n0 #density of PPC layer 2

    omega_pe_1 = smath.sqrt(n1*e*e/(eps0*me)) #plasma frequency of layer 1
    omega_pe_2 = smath.sqrt(n2*e*e/(eps0*me)) #plasma frequency of layer 2

    l = math.pi * c / (omega_l*smath.sqrt(1 - f0)) #periodic constant of PPC lambda
    h1 = l/2 #width of plasma layer 1, lambda/2 for simplicity
    h2 =  l/2 #width of plasma layer 2, lambda/2 for simplicity
    t1 = time.time()
    #print "Time to calculate constants = " + str(t1-t0) + "s"

#Calculating bands for the TE mode:
#@profile
def k_TE_func(omega,k_y):
    kx_1 = smath.sqrt(((omega**2)/(c**2)*((1-(omega_pe_1/omega)**2))) - (k_y**2)) #longitudinal incident plane wave vector x component
    kx_2 = smath.sqrt(((omega**2)/(c**2)*((1-(omega_pe_2/omega)**2))) - (k_y*k_y)) #longitudinal incident plane wave vector x component
    delta_TE = kx_1/kx_2  #simplifying parameter
    p1 = kx_1*h1 #simplifying parameter
    p2 = kx_2*h2 #simplifying parameter
    tr_u = 2*(((np.cos(p1))*(np.cos(p2))) - 0.5*(delta_TE + (1/delta_TE))*(np.sin(p1))*(np.sin(p2))) #trace of matrix U
    Kp = (1j/l)*(smath.log(0.5*tr_u + smath.sqrt(((0.5*tr_u)**2) - 1))) #Bloch Wave Vector, calculated with +
    Km = (1j/l)*(smath.log(0.5*tr_u - smath.sqrt(((0.5*tr_u)**2) - 1))) #Bloch Wave Vector, calculated with -
    t1 = time.time()
    return [Kp,Km]

#similar definitions hold for the TM mode calculations
#@profile
def k_TM_func(omega,k_y):
    kx_1 = smath.sqrt(((omega**2)/(c**2)*((1-(omega_pe_1/omega)**2))) - (k_y**2))
    kx_2 = smath.sqrt(((omega**2)/(c**2)*((1-(omega_pe_2/omega)**2))) - (k_y*k_y))
    delta_TM = (1-((omega_pe_2/omega)**2))/(1-((omega_pe_1/omega)**2))
    p1 = kx_1*h1
    p2 = kx_2*h2
    tr_u = 2*(((np.cos(p1))*(np.cos(p2))) - 0.5*(delta_TM + (1/delta_TM))*(np.sin(p1))*(np.sin(p2)))
    Kp = (1j/l)*(smath.log(0.5*tr_u + smath.sqrt(((0.5*tr_u)**2) - 1)))
    Km = 0
    Km = (1j/l)*(smath.log((0.5*tr_u) - smath.sqrt(((0.5*tr_u)**2) - 1)))
    return [Kp,Km]

#Evaluates K numerically for a range of values, and constructs a bitmap array with 1s for real K, 0s for complex
#@profile
def make_real_bitmap_TE(f0,f):
    t0 = time.time()
    calculate_constants(f0,f)
    xaxis = np.linspace(sys.float_info.min, 1.1, num=dim) #space of x coordinates
    yaxis = np.linspace(0.4,1.3, num=dim) #space of y coordinates
    normalised_k_y0,omega_0 = np.meshgrid(xaxis,yaxis)
    k_y =normalised_k_y0*2*math.pi/l
    evaluated_TE_p = k_TE_func(omega_0,k_y)[0]
    evaluated_TE_m = k_TE_func(omega_0,k_y)[1]
    TE_bitmap_bool_p = np.isreal(evaluated_TE_p)
    TE_bitmap_bool_m = np.isreal(evaluated_TE_p)
    TE_bitmap_bool = np.logical_or(TE_bitmap_bool_p,TE_bitmap_bool_m)
    TE_bitmap_bool = np.logical_not(TE_bitmap_bool)
    TE_bitmap = TE_bitmap_bool.astype(int)*255
    TE_bitmap= np.uint8(TE_bitmap)
    TE_bitmap = np.flipud(TE_bitmap)
    im_TE = Image.fromarray(TE_bitmap)
    t1 = time.time()
    #print "Time to make TE bitmap = " + str(t1-t0) + "s"
    return im_TE


#similar evaluation for TM mode
#@profile
def make_real_bitmap_TM(f0,f):
    t0 = time.time()
    calculate_constants(f0,f)
    xaxis = np.linspace(-sys.float_info.min, -1.1, num=dim) #space of x coordinates
    yaxis = np.linspace(0.4,1.3, num=dim) #space of y coordinates
    normalised_k_y0,omega_0 = np.meshgrid(xaxis,yaxis)
    k_y =normalised_k_y0*2*math.pi/l
    evaluated_TM_p = k_TM_func(omega_0,k_y)[0]
    evaluated_TM_m = k_TM_func(omega_0,k_y)[1]
    TM_bitmap_bool_p = np.isreal(evaluated_TM_p)
    TM_bitmap_bool_m = np.isreal(evaluated_TM_p)
    TM_bitmap_bool = np.logical_or(TM_bitmap_bool_p,TM_bitmap_bool_m)
    TM_bitmap_bool = np.logical_not(TM_bitmap_bool)
    TM_bitmap = TM_bitmap_bool.astype(int)*255
    #scaleTo255WV(TM_bitmap)

    TM_bitmap = np.uint8(TM_bitmap)
    TM_bitmap = np.fliplr(TM_bitmap)
    TM_bitmap = np.flipud(TM_bitmap)

    im_TM = Image.fromarray(TM_bitmap)
    t1 = time.time()
    #print "Time to make TM bitmap = " + str(t1-t0) + "s"
    return im_TM


def scaleTo255WV(arr):
	rows,cols = arr.shape
	code = """
		int Pos;
		for(int Row = 0; Row < rows; Row++)
		{
			for(int Col = 0; Col < cols; Col++)
			{
				Pos = Row*cols + Col;
				arr[Pos] = arr[Pos]*255;
		return_val = 1;
	"""
	res = wv.inline(code, ['arr','cols','rows'], headers = ['<math.h>'], compiler = 'gcc')
	return arr

#image concatenation utility function
#@profile
def concat_images_horizontally(im1,im2):
    t0 = time.time()
    images = [im1,im2]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
      t1 = time.time()
    #print "Time to concat = " + str(t1-t0) + "s"
    return new_im

#@profile
def imageForValues(f0,f):
    im_TE = make_real_bitmap_TE(f0, f)
    im_TM = make_real_bitmap_TM(f0, f)

    #Converting images to RGB, in preparation for export to PNG
    im_TE = im_TE.convert('RGB')
    im_TM = im_TM.convert('RGB')

    #Image export to PNG file
    #im_TE.save('bandgaps_TE.png')
    #im_TM.save('bandgaps_TM_2.png')

    #Final joined image of band structure
    final_im = concat_images_horizontally(im_TM,im_TE)
    return final_im
    #final_im.save('bandgaps_concat_2.png')

t0 = time.time()
for f0_val, f_val in frequency_values:
    im = imageForValues(f0_val,f_val)
    im.show()
    #im.save('bandgaps (f0 = ' + str(f0_val) + ' | f = ' + str(f_val) + ').png')
t1 = time.time()
print "New Runtime = "+ str(t1-t0)
