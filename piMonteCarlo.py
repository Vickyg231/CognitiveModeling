import numpy 
inCircle = 0
outCircle = 0
for i in range(100000):
    x = numpy.random.uniform(-1,1)
    y = numpy.random.uniform(-1,1)
    if ((x**2+y**2) <= 1):
        inCircle += 1
    else:
        outCircle += 1
piVal = 4 * (inCircle/(inCircle+outCircle))

print(piVal)