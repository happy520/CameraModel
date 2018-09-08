import math
import numpy as np
import cv2
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


def decomposeTransferMatrix(transferMatrix):
    intrinsicMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(
        transferMatrix)

    intrinsicMatrix = intrinsicMatrix * (1 / intrinsicMatrix[2][2])

    transVect = transVect * (1.0 / transVect[3])
    transVect = transVect[:3]
    transVect = -1 * np.dot(rotMatrix, transVect)
    extrinsicMatrix = np.hstack((rotMatrix, transVect))
    extrinsicMatrix = np.vstack((extrinsicMatrix, np.array([0, 0, 0, 1])))

    return intrinsicMatrix, extrinsicMatrix


def imageToPatient(intrinsicMatrix, extrinsicMatrix, focalLength, pt2dImage):
    pt2d_pic = np.append(pt2dImage, 1)
    pt2d_ccd = np.dot(np.linalg.inv(intrinsicMatrix), pt2d_pic)
    pt3d_ccd = focalLength * pt2d_ccd
    pt3d_ccd = np.append(pt3d_ccd, 1)
    pt3d_patient = np.dot(np.linalg.inv(extrinsicMatrix), pt3d_ccd)

    #test
    origin = np.append(pt2dImage, [0, 1])
    
    m1 = np.array([
        1,0,0,0,
        0,1,0,0,
        0,0,1,1,
        0,0,0,1
    ]).reshape(4,4)

    m2 = np.hstack((np.linalg.inv(intrinsicMatrix), np.array([0,0,0]).reshape(3,1)))
    m2 = np.vstack((m2, np.array([0,0,0,1])))

    m3 = np.array([
        focalLength,0,0,0,
        0,focalLength,0,0,
        0,0,focalLength,0,
        0,0,0,1
    ]).reshape(4,4)

    m4 = np.linalg.inv(extrinsicMatrix)

    tmp1 = np.dot(m2,m1)
    tmp2 = np.dot(m3, tmp1)
    tmp3 = np.dot(m4, tmp2)

    result = np.dot(tmp3, origin)

    return pt3d_patient[:3]


# test data from TiControl
tm_pic1 = np.array([
    1.1236559197, 1.1840254584, 7.0739501020, 247.6169398382,
    2.1258435195, 6.7489143248, -1.4468024637, 878.1753008221,
    0.0014758781, -0.0003709535, -0.0000098948, 1.0000000000]).reshape(3, 4)

tm_pic2 = np.array([
    0.5679531142, 1.6153339745, 5.6165747476, 267.0922841687,
    -5.5689968070, 1.8119169523, 0.0613561327, 712.0750844756,
    0.0002835329, 0.0011821867, -0.0002462334, 1.0000000000]).reshape(3, 4)

inPt2d_pic1 = np.array([406.1818181818, 919.8181818182])
inPt2d_pic2 = np.array([396.0465031813, 667.3024582120])
inPt3d = np.array([10.1273997185, 8.8289417193, 19.9876963525])

outPt2d_pic1 = np.array([569.8181818182, 306.1818181818])
outPt2d_pic2 = np.array([543.6336565387, 548.7056385498])
outPt3d = np.array([15.4294454000, -74.5765221499, 59.5944163533])

# other parameters
focal_length = -1500.0
pic_height = 1024
pic_width = 1024

# decompose transfer matrix
intrinsicMatrix1, extrinsicMatrix1 = decomposeTransferMatrix(tm_pic1)
intrinsicMatrix2, extrinsicMatrix2 = decomposeTransferMatrix(tm_pic2)

# calculate optical center in patient coordinate system
optical_center = np.array([0, 0, 0, 1])
optical_center1_patient = np.dot(
    np.linalg.inv(extrinsicMatrix1), optical_center)
optical_center2_patient = np.dot(
    np.linalg.inv(extrinsicMatrix2), optical_center)

# calculate image corner in patient coordinate system
pt2d_BL = np.array([0, 0])
pt2d_BR = np.array([pic_width, 0])
pt2d_TL = np.array([0, pic_height])
pt2d_TR = np.array([pic_width, pic_height])

pt3d_BL1_patient = imageToPatient(
    intrinsicMatrix1, extrinsicMatrix1, focal_length, pt2d_BL)
pt3d_BR1_patient = imageToPatient(
    intrinsicMatrix1, extrinsicMatrix1, focal_length, pt2d_BR)
pt3d_TL1_patient = imageToPatient(
    intrinsicMatrix1, extrinsicMatrix1, focal_length, pt2d_TL)
pt3d_TR1_patient = imageToPatient(
    intrinsicMatrix1, extrinsicMatrix1, focal_length, pt2d_TR)

pt3d_BL2_patient = imageToPatient(
    intrinsicMatrix2, extrinsicMatrix2, focal_length, pt2d_BL)
pt3d_BR2_patient = imageToPatient(
    intrinsicMatrix2, extrinsicMatrix2, focal_length, pt2d_BR)
pt3d_TL2_patient = imageToPatient(
    intrinsicMatrix2, extrinsicMatrix2, focal_length, pt2d_TL)
pt3d_TR2_patient = imageToPatient(
    intrinsicMatrix2, extrinsicMatrix2, focal_length, pt2d_TR)

# calculate back projection ray in patient coordinate system
inPt3d_pic1 = imageToPatient(
    intrinsicMatrix1, extrinsicMatrix1, focal_length, inPt2d_pic1)
inPt3d_pic2 = imageToPatient(
    intrinsicMatrix2, extrinsicMatrix2, focal_length, inPt2d_pic2)

outPt3d_pic1 = imageToPatient(
    intrinsicMatrix1, extrinsicMatrix1, focal_length, outPt2d_pic1)
outPt3d_pic2 = imageToPatient(
    intrinsicMatrix2, extrinsicMatrix2, focal_length, outPt2d_pic2)

# draw 3d patient coordinate system
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.array([pt3d_BR1_patient[0], pt3d_BL1_patient[0],
              pt3d_TR1_patient[0], pt3d_TL1_patient[0]]).reshape(2, 2)
y = np.array([pt3d_BR1_patient[1], pt3d_BL1_patient[1],
              pt3d_TR1_patient[1], pt3d_TL1_patient[1]]).reshape(2, 2)
z = np.array([pt3d_BR1_patient[2], pt3d_BL1_patient[2],
              pt3d_TR1_patient[2], pt3d_TL1_patient[2]]).reshape(2, 2)
ax.plot_surface(x, y, z, linewidth=0.2, antialiased=True, color='lightgrey')

x = np.array([pt3d_BR2_patient[0], pt3d_BL2_patient[0],
              pt3d_TR2_patient[0], pt3d_TL2_patient[0]]).reshape(2, 2)
y = np.array([pt3d_BR2_patient[1], pt3d_BL2_patient[1],
              pt3d_TR2_patient[1], pt3d_TL2_patient[1]]).reshape(2, 2)
z = np.array([pt3d_BR2_patient[2], pt3d_BL2_patient[2],
              pt3d_TR2_patient[2], pt3d_TL2_patient[2]]).reshape(2, 2)
ax.plot_surface(x, y, z, linewidth=0.2, antialiased=True, color='lightgrey')

x = np.array([optical_center1_patient[0], inPt3d_pic1[0]])
y = np.array([optical_center1_patient[1], inPt3d_pic1[1]])
z = np.array([optical_center1_patient[2], inPt3d_pic1[2]])
ax.plot(x, y, z)
x = np.array([optical_center2_patient[0], inPt3d_pic2[0]])
y = np.array([optical_center2_patient[1], inPt3d_pic2[1]])
z = np.array([optical_center2_patient[2], inPt3d_pic2[2]])
ax.plot(x, y, z)
ax.scatter(inPt3d[0], inPt3d[1], inPt3d[2],
           s=np.array([25]), c='r', marker='o')

x = np.array([optical_center1_patient[0], outPt3d_pic1[0]])
y = np.array([optical_center1_patient[1], outPt3d_pic1[1]])
z = np.array([optical_center1_patient[2], outPt3d_pic1[2]])
ax.plot(x, y, z)
x = np.array([optical_center2_patient[0], outPt3d_pic2[0]])
y = np.array([optical_center2_patient[1], outPt3d_pic2[1]])
z = np.array([optical_center2_patient[2], outPt3d_pic2[2]])
ax.plot(x, y, z)
ax.scatter(outPt3d[0], outPt3d[1], outPt3d[2],
           s=np.array([25]), c='b', marker='o')

ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')

plt.show()

# print information
print "intrinsicMatrix1"
print intrinsicMatrix1

print "extrinsicMatrix1"
print extrinsicMatrix1

print "intrinsicMatrix2"
print intrinsicMatrix2

print "extrinsicMatrix2"
print extrinsicMatrix2
