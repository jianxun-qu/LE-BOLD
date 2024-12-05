import SimpleITK as sitk

import matplotlib
import matplotlib.pyplot as plt

import os

import numpy as np

import scipy.io

import time


# Basic Function
def gen_dfm_zebra_2d(npa_in, period=4):
    dfm_npa = np.zeros(npa_in.shape)
    dfm_npa[::period, :] = 1
    dfm_npa[:, ::period] = 1
    return dfm_npa


# Start Time
start_time = time.time()

# Load Data

subj_folder = r"D:\Workspace_Jianxun\LE_Proj\LE_BOLD\le_bold\subj_TangFeiFei"

t2smap_filename = "t2s_map_R_1.mat"
t2sec1_filename = "t2s_ec1_R_1.mat"

t2smap_path = os.path.join(subj_folder, t2smap_filename)
t2sec1_path = os.path.join(subj_folder, t2sec1_filename)

t2sec1_mat = scipy.io.loadmat(t2sec1_path)
t2smap_mat = scipy.io.loadmat(t2smap_path)

t2sec1_npa = t2sec1_mat['t2s_ec1']
t2smap_npa = t2smap_mat['t2s_map']

phasenum = np.size(t2sec1_npa, 3)

# Export Data

outmat_filename = "t2s_reg_L_1.mat"
outmat_path = os.path.join(subj_folder, outmat_filename)

# Reference Image

t2sec1_ref_npa = t2sec1_npa[:, :, 0, 2]

dfm_zebra_npa = gen_dfm_zebra_2d(t2sec1_ref_npa)
dfm_zebra_img = sitk.GetImageFromArray(dfm_zebra_npa)

# plt.style.use('seaborn-bright')
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 1, 1)
# plt.imshow(dfm_zebra_npa, vmin=0, vmax=2)
# plt.pause(2)
# plt.show()

# plt.close()

# Loop

t2sec1_reg_npa = np.zeros(t2sec1_npa.shape)
t2smap_reg_npa = np.zeros(t2sec1_npa.shape)
dfm_zebra_reg_npa = np.zeros(t2sec1_npa.shape)

process_range = range(0, phasenum)
# process_range = [2, 3, 100]

for idx in process_range:

    print("**** Executing Phase: %d" % idx)

    t2sec1_tmp_npa = np.squeeze(t2sec1_npa[:, :, 0, idx])
    t2smap_tmp_npa = np.squeeze(t2smap_npa[:, :, 0, idx])

    fixedImaging = sitk.GetImageFromArray(t2sec1_ref_npa)
    movingImage = sitk.GetImageFromArray(t2sec1_tmp_npa)
    otherImage = sitk.GetImageFromArray(t2smap_tmp_npa)
    zebraImage = sitk.GetImageFromArray(dfm_zebra_npa)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImaging)
    elastixImageFilter.SetMovingImage(movingImage)

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("rigid"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))

    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()

    movingImageReg = elastixImageFilter.GetResultImage()

    # TransformixImageFilter -- Apply deformation to other
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.SetMovingImage(otherImage)
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.Execute()

    otherImageReg = transformixImageFilter.GetResultImage()

    # TransformixImageFilter -- Apply deformation to dfm_zebra
    transformixImageFilter.SetMovingImage(zebraImage)
    transformixImageFilter.Execute()

    zebraImageReg = transformixImageFilter.GetResultImage()

    t2sec1_reg_npa[:, :, 0, idx] = sitk.GetArrayFromImage(movingImageReg)
    t2smap_reg_npa[:, :, 0, idx] = sitk.GetArrayFromImage(otherImageReg)
    dfm_zebra_reg_npa[:, :, 0, idx] = sitk.GetArrayFromImage(zebraImageReg)

    # plt.clf()
    # plt.subplot(2, 3, 1)
    # plt.imshow(sitk.GetArrayFromImage(movingImageReg), vmin=0, vmax=1000, cmap='gray')
    # plt.subplot(2, 3, 2)
    # plt.imshow(sitk.GetArrayFromImage(otherImageReg), vmin=10, vmax=40, cmap='jet')
    # plt.subplot(2, 3, 3)
    # plt.imshow(sitk.GetArrayFromImage(zebraImageReg), vmin=0, vmax=2, cmap='jet')
    # plt.subplot(2, 3, 4)
    # plt.imshow(t2sec1_tmp_npa-t2sec1_ref_npa, vmin=-50, vmax=50, cmap='bwr')
    # plt.subplot(2, 3, 5)
    # plt.imshow(t2sec1_reg_npa[:, :, 0, idx] - t2sec1_ref_npa, vmin=-50, vmax=50, cmap='PiYG')

    # plt.pause(10)
    # plt.close()

# Export Data
outmat_mdict = {}
outmat_mdict['t2s_ec1'] = t2sec1_npa
outmat_mdict['t2s_map'] = t2smap_npa
outmat_mdict['t2s_ec1_reg'] = t2sec1_reg_npa
outmat_mdict['t2s_map_reg'] = t2smap_reg_npa
outmat_mdict['dfm_zebra_reg'] = dfm_zebra_reg_npa

scipy.io.savemat(outmat_path, mdict=outmat_mdict)


# Plot Figure

# plt.figure(2, figsize=(10, 5))

# for idx in process_range:
#     plt.subplot(1, 2, 1)
#     plt.imshow(t2sec1_npa[:, :, 0, idx] - t2sec1_npa[:, :, 0, 2], vmin=-100, vmax=100, cmap='bwr')
#     plt.subplot(1, 2, 2)
#     plt.imshow(t2sec1_reg_npa[:, :, 0, idx] - t2sec1_reg_npa[:, :, 0, 2], vmin=-100, vmax=100, cmap='bwr')
#     plt.pause(1)
#     plt.clf()

print("**** Total Execution Time is: %f" % (time.time() - start_time) )