# %%
# FINDING OUTLIER IMAGE SLICES
#####################################################################################################################
# destination = 'C:\\Users\\BGAdmin\\data\\MD816\\outliers'
# file_list = os.listdir(target_name)
# height = []
# width = []
# for j in file_list:
#     if '.tif' in j:
#         J = plt.imread(os.path.join(target_name,j))
#         height.append(J.shape[0])
#         width.append(J.shape[1])
#         ratio = J.shape[1]/J.shape[0]
#         shape_sum = J.shape[1]+J.shape[0]
#         if shape_sum >= 2308:
#             print('size > 99th:\n',j)
#             fname, ext = os.path.splitext(j)
#             plt.imsave(fname=os.path.join(destination, 'large', fname+'.png'), arr=J, format='png')
#         if ratio >= 1.66750:
#             print('ratio > 99th:\n', j)
#             plt.imsave(fname=os.path.join(destination, 'wide', fname+'.png'), arr=J, format='png')
#         if ratio <= 0.75074:
#             print('ratio < 1st:\n', j)
#             plt.imsave(fname=os.path.join(destination, 'tall', fname+'.png'), arr=J, format='png')


# %%
# size_ratio = np.array(width)/np.array(height)
# size_sum = np.array(width) + np.array(height)
# ratio_99 = np.percentile(size_ratio, 99)
# ratio_1 = np.percentile(size_ratio, 1)
# sum_99 = np.percentile(size_sum, 99)
# width_99 = np.percentile(width,99)
# height_99 = np.percentile(height,99)
# print('width 99th percentile: ', width_99)
# print('height 99th percentile: ', height_99)
# print('sum 99th percentile: ', sum_99)
# print('ratio 98% interval: ', ratio_1, ratio_99)

# %%
# fig, axs = plt.subplots(3)
# fig.set_size_inches(10,10)
# size_ratio = np.array(width)/np.array(height)
# axs[0].hist(size_ratio, bins=50)
# axs[0].set_title('Size Ratio (width/height)')
# axs[1].hist(height, bins=50)
# axs[1].set_title('Height')
# axs[2].hist(width, bins=50)
# axs[2].set_title('Width')
# plt.savefig(os.path.join(destination, 'histograms.png'))
# plt.show()

# %%
# import emlddmm
# from skimage import filters, measure
# import numpy as np
# from mayavi import mlab
# # from medpy.metric import binary
# import matplotlib
# matplotlib.use('qtagg')


# img = 'C:\\Users\\BGAdmin\\data\\MD816/HR_NIHxCSHL_50um_14T_M1_masked.vtk'
# out = 'C:\\Users\\BGAdmin\\emlddmm\\test_outputs'

# xI,I,_,_ = emlddmm.read_data(img)

# fig = emlddmm.draw(I,xI)
# fig[0].suptitle('image')
# fig[0].canvas.draw()

# dI = [x[1]-x[0] for x in xI]
# thresh = filters.threshold_otsu(np.array(I)[0,int(I.shape[1]/2),:,:])
# # AphiI_verts, AphiI_faces, AphiI_normals, AphiI_values  = measure.marching_cubes(np.squeeze(np.array(AphiI)), thresh, spacing=dI)
# I_verts, I_faces, I_normals, I_values = measure.marching_cubes(np.squeeze(np.array(I)), thresh, spacing=dI)

# surface_fig = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# # mlab.triangular_mesh(AphiI_verts[:,0], AphiI_verts[:,1], AphiI_verts[:,2], AphiI_faces, colormap='hot', opacity=0.5, figure=surface_fig)
# mlab.triangular_mesh(I_verts[:,0], I_verts[:,1], I_verts[:,2], I_faces, colormap='cool', opacity=0.5, figure=surface_fig)
# mlab.show()
# mlab.savefig(out+'surfaces.obj')
# mlab.close()

# # %%
# import numpy as np
# from skimage import measure, color, filters
# import emlddmm
# import matplotlib.pyplot as plt
# from mayavi import mlab


# img_dir = 'MD787_small_nissl'

# # visualize 3d mesh of 2d reconstructed volume
# xJ, J, _, _ = emlddmm.read_data(img_dir)
# mask = J[3,...]
# J_gray = color.rgb2gray(np.transpose(J[:3, ...]*mask, (1,2,3,0)))

# dJ = [x[1]-x[0] for x in xJ]
# thresh = filters.threshold_otsu(J_gray[int(J_gray.shape[0]/2), 50:250, 10:300])
# J_verts, J_faces, J_normals, J_values = measure.marching_cubes(np.squeeze(np.array(J_gray)), thresh, spacing=dJ)

# surface_fig = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(J_verts[:,0], J_verts[:,1], J_verts[:,2], J_faces, colormap='cool', opacity=0.5, figure=surface_fig)
# mlab.show()


# # %%

# idx = int(J_gray.shape[0]/2)
# J_slice = J_gray[idx, 50:250, 10:300]  # manually cropping image to exclude padding
# plt.figure()
# plt.imshow(J_slice, cmap = plt.cm.gray)

# thresh = filters.threshold_otsu(J_slice)

# contours = measure.find_contours(J_slice, thresh)

# # manually get the right contour
# contours_len = [len(c) for c in contours]
# contours_len[40] = 0
# contours_len_max_idx = np.argmax(contours_len)
# print(contours_len_max_idx)
# print(contours_len[contours_len_max_idx])
# # Display the image and plot all contours found
# fig, ax = plt.subplots()
# ax.imshow(J_slice, cmap=plt.cm.gray)

# for contour in contours:
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

# fig, ax = plt.subplots()
# ax.imshow(J_slice, cmap=plt.cm.gray)   
# ax.plot(contours[contours_len_max_idx][:, 1], contours[contours_len_max_idx][:, 0], linewidth=2)

# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()

# # %%
# # display binarized image using otsu's method

# idx = int(J_gray.shape[0]/2)
# J_slice = J_gray[idx, 50:250, 10:300]  # manually cropping image to exclude padding
# plt.figure()
# plt.imshow(J_slice, cmap = plt.cm.gray)

# thresh = filters.threshold_otsu(J_slice)
# # binarize each image
# J_slice_bool = (J_slice > thresh) * 1
# plt.figure()
# plt.imshow(J_slice_bool)

# # %%
# from skimage.filters import gaussian
# from skimage.segmentation import active_contour

# def get_border(img):
# # input: 2d image array
#     h = img.shape[0]-11
#     w = img.shape[1]-11
#     s1 = np.array([np.linspace(10,h,20), np.ones(20)*10]).T
#     s2 = np.array([(h)*np.ones(20), np.linspace(10,w,20)]).T[1:]
#     s3 = np.array([np.linspace(10,h,20)[::-1], w*np.ones(20)]).T[1:]
#     s4 = np.array([np.ones(20)*10, np.linspace(10,w,20)[::-1]]).T[1:]

#     return np.concatenate((s1,s2,s3,s4))

# img_dir = 'MD787_small_nissl'
# xJ, J, _, _ = emlddmm.read_data(img_dir)
# J_gray = color.rgb2gray(np.transpose(J[:3, ...], (1,2,3,0)))
# center = [int(J.shape[1]/2), int(J.shape[2]/2), int(J.shape[3]/2)]

# # plt.figure()
# # plt.imshow(J_gray[center[0]], cmap='gray')
# # plt.show()
# # b = get_border(J_gray[center[0]])

# w = []
# h = []
# W = []
# H = []
# J_cropped = []
# init = []
# snake = []
# for i in range(J.shape[1]):  # for each slice
#     row = np.where(mask[i,center[1],:]==True)
#     w.append(np.min(row))
#     W.append(np.max(row))
#     col = np.where(mask[i,:,center[2]]==True)
#     h.append(np.min(col))
#     H.append(np.max(col))
#     J_cropped.append(J_gray[i, h[i]:H[i], w[i]:W[i]])
#     init.append(get_border(J_cropped[i]))
#     snake.append( active_contour( J_cropped[i], #gaussian(J_cropped[i], 1, preserve_range=False),
#                                   init[i], boundary_condition='periodic',
#                                   alpha=0.0015, beta=10.0, w_line=-0.1, w_edge=2, gamma=0.001))
#     print(f'finished slice {i}')

# img = J_cropped[center[0]]
# fig, ax = plt.subplots(figsize=(9, 5))
# ax.imshow(img, cmap=plt.cm.gray)
# ax.plot(init[center[0]][:, 1], init[center[0]][:, 0], '--r', lw=3)
# ax.plot(snake[center[0]][:, 1], snake[center[0]][:, 0], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])

# plt.show()



# # %%
# plt.figure(figsize=(9, 5))
# plt.imshow(J_cropped[center[0]],cmap='gray')
# plt.show()


# %%
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE
import numpy as np
import emlddmm
import torch
import matplotlib.pyplot as plt

MR_img = '/home/brysongray/data/MD816_mini/HR_NIHxCSHL_50um_14T_M1_masked.vtk'
CCF_img = '/home/brysongray/data/MD816_mini/average_template_50.vtk'
# CCFtoMRI_disp = '/home/brysongray/emlddmm/transformation_graph_outputs/CCF/MRI_to_CCF/transforms/CCF_to_MRI_displacement.vtk'
# velocity = '/home/brysongray/emlddmm/transformation_graph_outputs/CCF/MRItoCCF/transforms/velocity.vtk'

xJ,J,title,names = emlddmm.read_data(MR_img)
xJ = [torch.as_tensor(x) for x in xJ]
# xI, I, title, names = emlddmm.read_data(CCF_img)

XJ = torch.stack(torch.meshgrid(xJ))
print(XJ.shape)
print(XJ[1:].permute(1,2,3,0).shape)
XJ_ = torch.stack(torch.meshgrid(xJ), -1)
print(XJ_[..., 1:].shape)
# x, disp, title, names = emlddmm.read_vtk_data(CCFtoMRI_disp)
# disp = torch.as_tensor(disp)

# down = [a//b for a, b in zip(I.shape[1:], disp.shape[2:])]

# xI,I = emlddmm.downsample_image_domain(xI,I,down)
# XI = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI]))

# X = disp[0] + XI

# AphiI = emlddmm.apply_transform_float(xJ, J, X)
# print(AphiI.shape)
# src='MRI'
# dest='CCF'
# fig = emlddmm.draw(AphiI, xI)
# fig[0].suptitle('transformed {src} to {dest}'.format(src=src, dest=dest))
# fig[0].canvas.draw()
# plt.show()

#%%
import os
import emlddmm
import torch
import numpy as np

out = '/home/brysongray/emlddmm/transformation_graph_outputs/'
space = "HIST"
hist_img = '/home/brysongray/data/MD816_mini/MD816_STIF_mini'
xJ,J,title,names = emlddmm.read_data(hist_img)
xJ = [torch.as_tensor(x) for x in xJ]
transforms = os.path.join(out, f'{space}_REGISTERED/{space}_INPUT_to_{space}_REGISTERED/transforms')
transforms_ls = sorted(os.listdir(transforms), key=lambda x: x.split('_matrix.txt')[0][-4:])

A2d = []
for t in transforms_ls:
    A2d_ = np.genfromtxt(os.path.join(transforms, t), delimiter=',')
    # note that there are nans at the end if I have commas at the end
    if np.isnan(A2d_[0, -1]):
        A2d_ = A2d_[:, :A2d_.shape[1] - 1]
    A2d.append(A2d_)
A2d = torch.as_tensor(np.stack(A2d))
A2di = torch.inverse(A2d)

print("A2di shape: ", A2di.shape)

#%%
# what I'm doing now
XJ = torch.stack(torch.meshgrid(xJ), -1) # easier to stack along dim -1 for my application
XJ_ = torch.clone(XJ.permute(3,0,1,2))
print("XJ shape: ", XJ.shape)
print("XJ_ shape before: ", XJ_.shape)
# print('A2di[:,None,None,:2,:2] shape: ', A2di[:,None,None,:2,:2].shape)
# print('(XJ_[1:].permute(1,2,3,0)[...,None]) shape: ', (XJ_[1:].permute(1,2,3,0)[...,None]).shape)
# print('A2di[:,None,None,:2,-1] shape: ', A2di[:,None,None,:2,-1].shape)
XJ_[1:] = ((A2di[:,None,None,:2,:2]@ (XJ_[1:].permute(1,2,3,0)[...,None]))[...,0] + A2di[:,None,None,:2,-1]).permute(3,0,1,2)  

print('XJ_ shape after: ', XJ_.shape)
#%%
# what write qc does
XJ = torch.stack(torch.meshgrid(xJ)) # stack along dim 1
print('XJ shape: ', XJ.shape)

XJ_ = torch.clone(XJ)
print('XJ_ shape before: ', XJ_.shape)
XJ_[1:] = ((A2di[:,None,None,:2,:2]@ (XJ[1:].permute(1,2,3,0)[...,None]))[...,0] + A2di[:,None,None,:2,-1]).permute(3,0,1,2)
print('XJ_ shape: ', XJ_.shape)

#%%
# better way. Stack along dim -1 with fewer permutations
XJ = torch.stack(torch.meshgrid(xJ), -1)
XJ_ = torch.clone(XJ)
print("XJ shape: ", XJ.shape)
print("XJ_ shape before: ", XJ_.shape)

XJ_[..., 1:] = ((A2di[:,None,None,:2,:2]@ (XJ_[..., 1:][...,None]))[...,0] + A2di[:,None,None,:2,-1])
print('XJ_ shape after: ', XJ_.shape)

#%%
import emlddmm
import matplotlib.pyplot as plt

# img = '/home/brysongray/emlddmm/transformation_graph_outputs/HIST_INPUT/HIST_REGISTERED_to_HIST_INPUT/images/HIST_REGISTERED_MD816-N1-2021.04.05-17.44.55_MD816_1_0001_to_HIST_INPUT_MD816-N1-2021.04.05-17.44.55_MD816_1_0001.vtk'
img1 = '/home/brysongray/emlddmm/transformation_graph_outputs/MRI/HIST_INPUT_to_MRI/images/HIST_nissl_to_MRI.vtk'
img2 = '/home/brysongray/emlddmm/transformation_graph_outputs/CCF/MRI_to_CCF/images/MRI_masked_to_CCF.vtk'
CT_to_MRI = '/home/brysongraylocal/emlddmm/transformation_graph_outputs/MRI/CT_to_MRI/images/CT_masked_to_MRI.vtk'
MRI_to_CT = '/home/brysongraylocal/emlddmm/transformation_graph_outputs/CT/MRI_to_CT/images/MRI_masked_to_CT.vtk'

xJ, J, title, names = emlddmm.read_data(CT_to_MRI)
xI, I, title, names = emlddmm.read_data(MRI_to_CT)

fig1 = emlddmm.draw(J,xJ)
fig1[0].canvas.draw()

fig2 = emlddmm.draw(I,xI)
fig2[0].canvas.draw()

plt.show()
# %%
import emlddmm

MRItoCCF = '/home/brysongray/emlddmm/transformation_graph_outputs/CCF/MRI_to_CCF/images/MRI_masked_to_CCF.vtk'
CCFtoMRI = '/home/brysongray/emlddmm/transformation_graph_outputs/MRI/CCF_to_MRI/images/CCF_average_template_50_to_MRI.vtk'
HISTtoMRI = '/home/brysongray/emlddmm/transformation_graph_outputs/MRI/HIST_to_MRI/images/HIST_nissl_to_MRI.vtk'
xI, I, _,_ = emlddmm.read_data(MRItoCCF)
xJ, J, _,_ = emlddmm.read_data(HISTtoMRI)
print(J.shape)
print([len(x) for x in xJ])
#%%
# emlddmm.draw(I,xI)
print(J.shape)
print(J[:,256,None,...].shape)
xJ_ = [np.array([xJ[0][256]]), xJ[1], xJ[2]]

emlddmm.draw(J[:,256,None, ...], xJ_)
plt.show()

# %%
# in the special case of transforming an image series to registered or input space, the dest will be a directory containing 2d Affines
if dest_path == f'{dest_space}_REGISTERED/{dest_space}_INPUT_to_{dest_space}_REGISTERED':
    xJ, J, J_title, _ = emlddmm.read_data(src_path) # the image to be transformed
    J = J.astype(float)
    J = torch.as_tensor(J,dtype=dtype,device=device)
    x_series = [torch.as_tensor(x,dtype=dtype,device=device) for x in xJ]
    X_series = torch.stack(torch.meshgrid(x_series), -1)
    transforms_ls = os.listdir(os.path.join(out, dest_path))
    transforms_ls = sorted(transforms_ls, key=lambda x: x.split('_matrix.txt')[0][-4:])

    A2d = []
    for t in transforms_ls:
        A2d_ = np.genfromtxt(os.path.join(out, dest_path, t), delimiter=',')
        # note that there are nans at the end if I have commas at the end
        if np.isnan(A2d_[0, -1]):
            A2d_ = A2d_[:, :A2d_.shape[1] - 1]
        A2d.append(A2d_)

    A2d = torch.as_tensor(np.stack(A2d),dtype=dtype,device=device)
    A2di = torch.inverse(A2d)
    points = (A2di[:, None, None, :2, :2] @ X_series[..., 1:, None])[..., 0] # reconstructed space needs to be created from the 2d series coordinates
    m0 = torch.min(points[..., 0])
    M0 = torch.max(points[..., 0])
    m1 = torch.min(points[..., 1])
    M1 = torch.max(points[..., 1])
    # construct a recon domain
    dJ = [x[1] - x[0] for x in x_series]
    # print('dJ shape: ', [x.shape for x in dJ])
    xr0 = torch.arange(float(m0), float(M0), dJ[1], device=m0.device, dtype=m0.dtype)
    xr1 = torch.arange(float(m1), float(M1), dJ[2], device=m0.device, dtype=m0.dtype)
    xr = x_series[0], xr0, xr1
    XR = torch.stack(torch.meshgrid(xr), -1)
    # reconstruct 2d series
    Xs = torch.clone(XR)
    Xs[..., 1:] = (A2d[:, None, None, :2, :2] @ XR[..., 1:, None])[..., 0] + A2d[:, None, None, :2, -1]
    Xs = Xs.permute(3, 0, 1, 2)
    Jr = emlddmm.interp(xJ, J, Xs)

    # write out displacement
    input_disp = (Xs - X_series.permute(3,0,1,2)).cpu()[None]
    for i in range(input_disp.shape[2]):
        x_series_ = [x_series[0][i], x_series[1], x_series[2]]
        x_series_[0] = torch.tensor([x_series_[0], x_series_[0] + 10])
        xr_ = [xr[0][i], xr[1], xr[2]]
        xr_[0] = torch.tensor([xr_[0], xr_[0] + 10])
        # write out input to dest displacement
        input_dir = os.path.join(out, f'{dest_space}_REGISTERED/{dest_space}_INPUT_to_{dest_space}_REGISTERED')
        output_name = os.path.join(input_dir, transforms_ls[i].split('_matrix.txt')[0]+'_displacement.vtk')
        title = transforms_ls[i].split('_matrix.txt')[0]+'_displacement'
        emlddmm.write_vtk_data(output_name, x_series_, input_disp[:,:, i, None, ...], title)

    # write out image
    fig = emlddmm.draw(Jr, xr)
    fig[0].suptitle(f'transformed {src_space} {src_img} to {dest_space} REGISTERED')
    fig[0].canvas.draw()
    # save transformed 3d image   
    img_out = os.path.join(out, f'{dest_space}_REGISTERED/{dest_space}_INPUT_to_{dest_space}_REGISTERED')
    if not os.path.exists(img_out):
        os.makedirs(img_out)
    emlddmm.write_vtk_data(os.path.join(img_out, f'{src_space}_INPUT_to_{dest_space}_REGISTERED.vtk'), xr, Jr, f'{src_space}_INPUT_to_{dest_space}_REGISTERED')

#%%
# transform images using disp fields
def transform_img(adj, spaces, src, dest, out, src_img='', dest_img=''):
    '''
    Parameters
    ----------
    adj: adjacency list
        of the form: [{1: ('path to transformation from space 0 to space 1', 'b'}, {0: ('path to transform from space 0 to space 1', 'f') }]
    spaces: spaces dict
        example: {'MRI':0, 'CT':1, 'ATLAS':2}
    src: source space (str)
    dest: destination space (str)
    out: output directory
    src_img: path to source image (image to be transformed)
    dest_img: path to destination image (image in space to which source image will be matched)
    
    Returns
    ----------
    x: List of arrays storing voxel locations
    AphiI: Transformed image as tensor

    input: path to image to be transformed (src_img or I), img space to to which the source image will be matched (dest_img, J),
     adjacency list and spaces dict from run_registration, source and destination space names
    return: x, transfromed image
    '''
    # get transformation sequence
    path = findShortestPath(adj, spaces[src], spaces[dest], len(spaces))
    if len(path) < 2:
        return
    print("\nPath is:")

    for i in path:
        for key, value in spaces.items():
            if i == value:
                print(key, end=' ')

    transformation_seq = getTransformation(adj, path)
    print('\nTransformation sequence: ', transformation_seq)

    # load source and destination images
    xI, I, I_title, _ = emlddmm.read_data(dest_img) # the space to transform into
    I = I.astype(float)
    I = torch.as_tensor(I, dtype=dtype, device=device)
    xI = [torch.as_tensor(x,dtype=dtype,device=device) for x in xI]
    xJ, J, J_title, _ = emlddmm.read_data(src_img) # the image to be transformed
    J = J.astype(float)
    J = torch.as_tensor(J,dtype=dtype,device=device)
    xJ = [torch.as_tensor(x,dtype=dtype,device=device) for x in xJ]

    # compose displacements
    # TODO
    # first interpolate all displacements in sequence into destination space,
    # then add them together
    disp_list = []
    for i in reversed(range(len(transformation_seq))):
        # path = glob.glob(transformation_seq[i] + '/transforms/*displacement.vtk')
        x, disp, title, names = emlddmm.read_vtk_data(transformation_seq[i])
        if i == len(transformation_seq)-1: # if the first displacement in the sequence
            down = [a//b for a, b in zip(I.shape[1:], disp.shape[2:])] # check if displacement image was downsampled from original
            xI,I = emlddmm.downsample_image_domain(xI,I,down) # downsample the domain
            XI = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xI]))
            disp_list.append(torch.as_tensor(disp)) # add the displacement
        else: # otherwise we need to interpolate to the destination space
            ID = torch.stack(torch.meshgrid(x))[None]
            disp = emlddmm.interp(x,(disp - ID), XI) + XI
            disp_list.append(torch.as_tensor(disp))
    # sum the displacements
    # v = torch.cat(disp_list)
    disp = torch.sum(torch.stack(disp_list),0).to(device=device)
    print('disp shape: ', disp.shape)
    print('XI shape: ', XI.shape)
    X = disp[0] + XI

    AphiI = emlddmm.apply_transform_float(xJ, J, X)

    # save figure and text file of transformation order
    if not os.path.exists(out):
        os.makedirs(out)
    # plt.savefig(os.path.join(out, '{src}_{dest}'.format(src=src, dest=dest))) # TODO: this image will be saved in qc
    with open(os.path.join(out, '{src}_{dest}.txt'.format(src=src, dest=dest)), 'w') as f:
        for transform in reversed(transformation_seq[1:]):
            f.write(str(transform) + ', ')
        f.write(str(transformation_seq[0]))

    return xI, AphiI
#%%
# Note: This code is from work on constructing 3d visualization and hausdorff based registration validation
######################################################################################################################################
if not slice_matching:
    Jr_ = Jr.cpu().numpy()[0]
    AphiI = AphiI.cpu().numpy()
    # find Hausdorff distance between transformed atlas and target
    dJ = [x[1]-x[0] for x in xr]
    thresh = filters.threshold_otsu(Jr_[int(Jr_.shape[0]/2), :, :])
    AphiI_verts, AphiI_faces, AphiI_normals, AphiI_values  = measure.marching_cubes(np.squeeze(AphiI), thresh, spacing=dJ)
    J_verts, J_faces, J_normals, J_values = measure.marching_cubes(np.squeeze(np.array(Jr_)), thresh, spacing=dJ)

    distances = np.sum((J_verts[::4][None] - AphiI_verts[::4][:,None])**2, axis=-1)
    distances = np.sqrt(np.min(distances, axis=1))
    count, bins_count = np.histogram(distances, bins=1000)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    hausdorff = np.max(distances)
    hausdorff95 = np.percentile(distances, 95)

    # plot cdf
    fig = plt.figure()
    plt.plot(bins_count[1:], cdf)
    plt.title('CDF')
    plt.savefig(output_dir+'cdf.png')
    plt.close()

    # Visualize surfaces in 3d and save in OBJ file
    surface_fig = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.triangular_mesh(AphiI_verts[:,0], AphiI_verts[:,1], AphiI_verts[:,2], AphiI_faces, colormap='hot', opacity=0.5, figure=surface_fig)
    mlab.triangular_mesh(J_verts[:,0], J_verts[:,1], J_verts[:,2], J_faces, colormap='cool', opacity=0.5, figure=surface_fig)
    mlab.savefig(output_dir+'surfaces.obj')
    mlab.close()
    
    with open(output_dir+'qc.txt', 'w') as f:
        f.write('Symmetric Hausdorff Distance: '+str(hausdorff)+'\n95th Percentile Hausdorff Distance: '+str(hausdorff95)+'\nDice Coefficient: '+str(dice_coeff))
    print('Hausdorff distance: ', hausdorff, '\nhd95: ', hausdorff95, '\ndice: ', dice_coeff)

else:
    # compute per-slice hausdorff and dice
    dJ = [x[1] - x[0] for x in xr[1:]]

    # initialize lists for hd, hd95 and dice for each slice
    hausdorff = []
    hausdorff95 = []
    dice_coeff = []
    n = 0
    for i in range(Jr_.shape[0]): # for each slice in the volume, Jr
        # get J slice (Jr is already in grayscale)
        J_slice = Jr_[i]
        # find thresh value using otsu's algorithm
        thresh = filters.threshold_otsu(J_slice)
        # binarize each image
        J_slice_bool = (J_slice > thresh) * 1
        AphiI_slice_bool = (AphiI[0, i, ...] > thresh) * 1
        if np.max(AphiI_slice_bool) == 0:  # if the array contains all zeros it will not work
            continue
        # get hd, hd95, and dice coeff
        hausdorff.append(binary.hd(J_slice_bool, AphiI_slice_bool, voxelspacing=dJ))
        hausdorff95.append(binary.hd95(J_slice_bool, AphiI_slice_bool, voxelspacing=dJ))
        dice_coeff.append(dice(J_slice_bool.flatten(), AphiI_slice_bool.flatten()))
        with open(os.path.join(output_dir, f'slice_{i:04d}.txt'), 'w') as f:
            f.write('Symmetric Hausdorff Distance: '+str(hausdorff[n])+'\n95th Percentile Hausdorff Distance: '
                    + str(hausdorff95[n])+'\nDice Coefficient: '+str(dice_coeff[n]))
        n += 1  

    hd_mean = np.mean(hausdorff)
    hd95_mean = np.mean(hausdorff95)
    dice_mean = np.mean(dice_coeff)
    print('Mean hausdorff distance: ', hd_mean, '\nMean hd95: ', hd95_mean, '\nMean dice: ', dice_mean)

    with open(output_dir + 'qc.txt', 'a') as f:
        f.write('\nMean Hausdorff Distance: ' + str(hd_mean) + '\nMean 95th Percentile Hausdorff Distance: ' + str(
            hd95_mean) + '\nMean Dice Coefficient: ' + str(dice_mean))

fig = draw(torch.cat((I,phiiAiJ,I),0),xI)
fig[0].suptitle('atlas space')
#################################################################################################################################

#%%
import pickle

adj = pickle.load(open('outputs/transformation_graph_4-6_outputs/adjacency_list.p', 'rb'))
spaces = pickle.load(open('outputs/transformation_graph_4-6_outputs/spaces_dict.p', 'rb'))

print(adj)
print(spaces)


# %%
import transformation_graph
#%%
# add edge example

spaces = {'HIST': 0, 'MRI': 1, 'CCF': 2, 'CT': 3}
adj = [{} for i in range(len(spaces))]

print(adj)

transformation_graph.add_edge(adj, spaces, 'MRI', 'CCF', 'outputs')
print(adj)

transformation_graph.add_edge(adj, spaces, 'HIST', 'MRI', 'outputs', slice_matching=True)
print(adj)

# transformation_graph.add_edge(adj, )
# %%
# BFS example

path = transformation_graph.find_shortest_path(adj, 0, 2, 4)
#%%
print('\n path: ', path)
# %%
path = transformation_graph.find_shortest_path(adj, 0, 3, 4)

# %%
# get_transformation example

transformation = transformation_graph.get_transformation(adj, path)
print(transformation)
# %%
# reg example
import transformation_graph

dest = '/home/brysongray/data/MD816_mini/average_template_50.vtk'
source  = '/home/brysongray/data/MD816_mini/HR_NIHxCSHL_50um_14T_M1_masked.vtk'
registration = [['MRI','masked'], ['CCF','average_template_50']]
config = 'configMD816_MR_to_CCF.json'
out = 'outputs/example_output'

transformation_graph.reg(dest, source, registration, config, out)


# %%
import torch
import numpy as np

def draw(J,xJ=None,fig=None,n_slices=5,vmin=None,vmax=None,disp=True,**kwargs):    
    """ Draw 3D imaging data.
    
    Images are shown by sampling slices along 3 orthogonal axes.
    Color or grayscale data can be shown.
    
    Parameters
    ----------
    J : array like (torch tensor or numpy array)
        A 3D image with C channels should be size (C x nslice x nrow x ncol)
        Note grayscale images should have C=1, but still be a 4D array.
    xJ : list
        A list of 3 numpy arrays.  xJ[i] contains the positions of voxels
        along axis i.  Note these are assumed to be uniformly spaced. The default
        is voxels of size 1.0.
    fig : matplotlib figure
        A figure in which to draw pictures. Contents of the figure will be cleared.
        Default is None, which creates a new figure.
    n_slices : int
        An integer denoting how many slices to draw along each axis. Default 5.
    vmin
        A minimum value for windowing imaging data. Can also be a list of size C for
        windowing each channel separately. Defaults to None, which corresponds 
        to tha 0.001 quantile on each channel.
    vmax
        A maximum value for windowing imaging data. Can also be a list of size C for
        windowing each channel separately. Defaults to None, which corresponds 
        to tha 0.999 quantile on each channel.
    kwargs : dict
        Other keywords will be passed on to the matplotlib imshow function. For example
        include cmap='gray' for a gray colormap

    Returns
    -------
    fig : matplotlib figure
        The matplotlib figure variable with data.
    axs : array of matplotlib axes
        An array of matplotlib subplot axes containing each image.


    """
    if type(J) == torch.Tensor:
        J = J.detach().clone().cpu()
    J = np.array(J)
    if xJ is None:
        nJ = J.shape[-3:]
        xJ = [np.arange(n) - (n-1)/2.0 for n in nJ] 
    if type(xJ[0]) == torch.Tensor:
        xJ = [np.array(x.detach().clone().cpu()) for x in xJ]
    xJ = [np.array(x) for x in xJ]
    
    if fig is None:
        fig = plt.figure()
    fig.clf()    
    if vmin is None:
        vmin = np.quantile(J,0.001,axis=(-1,-2,-3))
    if vmax is None:
        vmax = np.quantile(J,0.999,axis=(-1,-2,-3))
    vmin = np.array(vmin)
    vmax = np.array(vmax)    
    # I will normalize data with vmin, and display in 0,1
    if vmin.ndim == 0:
        vmin = np.repeat(vmin,J.shape[0])
    if vmax.ndim == 0:
        vmax = np.repeat(vmax,J.shape[0])
    if len(vmax) >= 2 and len(vmin) >= 2:
        # for rgb I'll scale it, otherwise I won't, so I can use colorbars
        J -= vmin[:,None,None,None]
        J /= (vmax[:,None,None,None] - vmin[:,None,None,None])
        J[J<0] = 0
        J[J>1] = 1
        vmin = 0.0
        vmax = 1.0
    # I will only show the first 3 channels
    if J.shape[0]>3:
        J = J[:3]
    if J.shape[0]==2:
        J = np.stack((J[0],J[1],J[0]))
    
    
    axs = []
    axsi = []
    # ax0
    slices = np.round(np.linspace(0,J.shape[1]-1,n_slices+2)[1:-1]).astype(int)        
    # for origin upper (default), extent is x (small to big), then y reversed (big to small)
    extent = (xJ[2][0],xJ[2][-1],xJ[1][-1],xJ[1][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1)
        toshow = J[:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax1
    slices = np.round(np.linspace(0,J.shape[2]-1,n_slices+2)[1:-1]).astype(int)    
    extent = (xJ[2][0],xJ[2][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):
        ax = fig.add_subplot(3,n_slices,i+1+n_slices)      
        toshow = J[:,:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    axsi = []
    # ax2
    slices = np.round(np.linspace(0,J.shape[3]-1,n_slices+2)[1:-1]).astype(int)        
    extent = (xJ[1][0],xJ[1][-1],xJ[0][-1],xJ[0][0])
    for i in range(n_slices):        
        ax = fig.add_subplot(3,n_slices,i+1+n_slices*2)
        toshow = J[:,:,:,slices[i]].transpose(1,2,0)
        if toshow.shape[-1] == 1:
            toshow = toshow.squeeze(-1)
        ax.imshow(toshow,vmin=vmin,vmax=vmax,aspect='equal',extent=extent,**kwargs)
        if i>0: ax.set_yticks([])
        axsi.append(ax)
    axs.append(axsi)
    
    fig.subplots_adjust(wspace=0,hspace=0)
    if not disp:
        plt.close(fig)

    return fig,axs
# %%
import emlddmm
import matplotlib.pyplot as plt
import numpy as np
img = '/home/brysongray/data/MD816_mini/average_template_50.vtk'
xJ, J, title, name = emlddmm.read_data(img)
J = J.astype('float64')
print(J.dtype)
print(type(J))
vmin = np.quantile(J,0.001,axis=(-1,-2,-3))
vmax = np.quantile(J,0.999,axis=(-1,-2,-3))
vmin = np.array(vmin)
vmax = np.array(vmax)
print(vmax.dtype)
print(type(vmax))
vmax = np.array([0.0])
J -= vmin[:,None,None,None]
J /= (vmax[:,None,None,None] - vmin[:,None,None,None])
#%%
# fig = draw(J,xJ,disp=False)
f,ax = plt.subplots()
ax.cla()
ax.imshow(J[0][132])
plt.close()
# plt.close()
# fig[0].savefig('outputs/savefig_test')
# f = draw(J,xJ, fig)
# %%
import transformation_graph

# run_registration example

reg_list = [{'registration':[['MRI','masked'],['CCF','average_template_50']],
             'source': '/home/brysongray/data/MD816_mini/HR_NIHxCSHL_50um_14T_M1_masked.vtk',
             'dest': '/home/brysongray/data/MD816_mini/average_template_50.vtk',
             'config': 'examples/configMD816_MR_to_CCF.json',
             'output': 'outputs/example_output'},
            {'registration':[['HIST','Nissl'],['MRI','masked']],
             'source': '/home/brysongray/data/MD816_mini/MD816_STIF_mini',
             'dest': '/home/brysongray/data/MD816_mini/HR_NIHxCSHL_50um_14T_M1_masked.vtk',
             'config': 'examples/configMD816_Nissl_to_MR.json',
             'output': 'outputs/example_output'}]
adj,spaces = transformation_graph.run_registrations(reg_list)
# %%
import emlddmm
import numpy as np

img = '/home/brysongray/data/MD816_mini/HR_NIHxCSHL_50um_14T_M1_masked.vtk'
# img = '/home/brysongray/data/MD816_mini/average_template_50.vtk'
disp_path = '/home/brysongray/emlddmm/outputs/transformation_graph_4-6_outputs/MRI/CCF_to_MRI/transforms/MRI_to_CCF_displacement.vtk'
xI, I, name, title = emlddmm.read_data(img)
_,disp,_,_ = emlddmm.read_data(disp_path)

dv = [(x[1]-x[0]) for x in xI]


#%%
grad = np.gradient(disp[0,0], dv[0],dv[1],dv[2])#, axis=(-1,-2,-3))
print([x.shape for x in grad])

# %%
# jacobian = lambda disp,dv : np.stack((np.stack(np.gradient(disp[0,0], dv[0], dv[1], dv[2]), axis=-1), 
#                                       np.stack(np.gradient(disp[0,1], dv[0], dv[1], dv[2]), axis=-1),
#                                       np.stack(np.gradient(disp[0,2], dv[0], dv[1], dv[2]), axis=-1)), axis=-1)

# J = jacobian(disp,dv)

jacobian2 = lambda X,dv : np.stack(np.gradient(X, dv[2],dv[1],dv[0], axis=(1,2,3))).transpose(2,3,4,0,1)

J2 = jacobian2(disp[0],dv)
detjac = np.linalg.det(J2)
# print(np.allclose(J,J2))

#%%
import emlddmm
import nibabel as nib
import nibabel.processing
import skimage
import numpy as np
#%%
target_f = '/home/brysongray/emlddmm/tests/194062_red_mm_SLA.nii.gz'
template_f = '/home/brysongray/emlddmm/tests/average_template_25_mm_ASL.nii.gz'
J_ = nib.load(target_f)
J = J_.get_fdata()
I_ = nib.load(template_f)
I = I_.get_fdata()


#%%
d = 8
Jd = skimage.transform.resize(J, (J.shape[0]//d, J.shape[1]//d, J.shape[2]//d), anti_aliasing=True)
Id = skimage.transform.resize(I, (I.shape[0]//d, I.shape[1]//d, I.shape[2]//d), anti_aliasing=True)
#%%
Jdiv = np.array([J.shape[0]/Jd.shape[0], J.shape[1]/Jd.shape[1], J.shape[2]/Jd.shape[2]])
Idiv = np.array([I.shape[0]/Id.shape[0], I.shape[1]/Id.shape[1], I.shape[2]/Id.shape[2]])
J_.header["pixdim"][1:4] = J_.header["pixdim"][1:4] * Jdiv
I_.header["pixdim"][1:4] = I_.header["pixdim"][1:4] * Idiv
#%%
Jout = nib.Nifti1Image(Jd, J_.affine, J_.header)
Iout = nib.Nifti1Image(Id, I_.affine, I_.header)
nib.save(Jout, '/home/brysongray/emlddmm/tests/194062_red_mm_SLA_down.nii' )
nib.save(Iout, '/home/brysongray/emlddmm/tests/average_template_25_mm_ASL_down.nii')

#%%
target_f = '/home/brysongray/emlddmm/tests/194062_red_mm_SLA_down.nii'
template_f = '/home/brysongray/emlddmm/tests/average_template_25_mm_ASL_down.nii'
Jd_ = nib.load(target_f)
Id_ = nib.load(template_f)
Jd = Jd_.get_fdata()
# %%
nib.save(Jd_, '/home/brysongray/emlddmm/tests/194062_red_mm_SLA_down.nii.gz' )
nib.save(Id_, '/home/brysongray/emlddmm/tests/average_template_25_mm_ASL_down.nii.gz')
# %%
import emlddmm
import numpy as np
#%%
ni = 120
nj = 120
nk = 120
xI = [np.arange(ni)-(ni-1)/2,np.arange(nj)-(nj-1)/2,np.arange(nk)-(nk-1)/2]
XI = np.stack(np.meshgrid(xI[0],xI[1],xI[2], indexing='ij'))
# condition is the surface of an ellipsoid with axes a, b, c
condition = lambda x,a,b,c : x[0]**2 / a**2 + x[1]**2 / b**2 + x[2]**2 / c**2
a = 15
b = 30
c = 20
v = np.where(condition(XI,a,b,c) <= 1.0, 1.0, 0.0)
#%%
import time
# write out ellipsoid image
fname = '/home/brysongray/emlddmm/tests/ellipsoid_img.vtk'
title = 'ellipsoid'
emlddmm.write_vtk_data(fname, xI, v[None], title)
writetime = os.path.getmtime('/home/brysongray/emlddmm/tests/ellipsoid_img.vtk')
print(round(writetime, 0)==round(time.time(),0))
# %%
import matplotlib.pyplot as plt
fname = '/home/brysongray/emlddmm/tests/ellipsoid_img.vtk'

xI, I, _,_ = emlddmm.read_vtk_data(fname)
fig = plt.figure(figsize=(10,10))
emlddmm.draw(I,xI,fig, n_slices=8, cmap='gray')

# %%
print(np.allclose(I,v[None]))
# %%
