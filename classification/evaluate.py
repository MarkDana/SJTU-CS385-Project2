# @MarkDana, 20200616
import os, random
import numpy as np
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from mpl_toolkits.mplot3d import axes3d, Axes3D
from subprocess import call
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable as V
from sklearn.decomposition import PCA, IncrementalPCA
from models import *
import dataloader
import torch.backends.cudnn as cudnn
import re
import gradcam
import cv2

NETWRKS = {'resnet18': ResNet18, 'resnet34': ResNet34, 'resnet50': ResNet50, 'resnet101': ResNet101, 'resnet152': ResNet152,
           'vgg11': VGG11, 'vgg13': VGG13, 'vgg16': VGG16, 'vgg19': VGG19, }
FEATURE_NAMES = {'resnet50': ['layer1','layer2','layer3','layer4'],
                 'vgg16': ['features.6', 'features.13', 'features.23', 'features.33', 'features.43']}
FEATURE_MODULES = {'resnet50': lambda x:x.layer4, 'vgg16': lambda x:x.features}
TARGET_LAYER_NAMES = {'resnet50': ["2"], 'vgg16': ["44"]}

NTWK_FT_DIMS = {'resnet50':{'layer1':64, 'layer4':512, 'layer3':256, 'layer2':128, 'out':10}, 'vgg16': {'features.6': 64, 'features.33': 512, 'features.23': 256, 'features.13': 128, 'out': 10, 'features.43': 512}}
PICK_FILTERS = 10
PICK_FILTER_INDICES = {'resnet50': {'layer1': [5, 7, 14, 20, 25, 29, 39, 47, 52, 60], 'layer4': [77, 81, 114, 194, 196, 198, 284, 393, 453, 486], 'layer3': [8, 34, 60, 79, 101, 121, 134, 218, 228, 251], 'layer2': [36, 41, 72, 81, 101, 103, 105, 110, 112, 121], 'out': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}, 'vgg16': {'features.6': [3, 6, 8, 23, 25, 49, 50, 55, 58, 61], 'features.33': [26, 56, 92, 168, 214, 274, 312, 397, 401, 420], 'features.23': [3, 18, 33, 39, 136, 208, 212, 234, 244, 252], 'features.13': [9, 17, 59, 76, 77, 107, 110, 111, 119, 124], 'features.43': [24, 49, 178, 206, 241, 243, 294, 306, 350, 375], 'out': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}}
NUMS_PER_COL = 3
CHANNEL_PRINT = 3
WIDTH_PRINT = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def drawCurveDonkey(intxtpath, outimgpath, title, xlabel='epoch', par1label='loss', par2label='accuracy(%)'):
    # each line of intxtpath txt is x\tp1\tp2, eg. epoch\tloss\tacc
    # par1label for host, par2label for par1
    xs = []
    p1s = []
    p2s = []

    with open(intxtpath, 'r') as fin:
        lines = [l.strip() for l in fin.readlines()]
        for line in lines:
            x, p1, p2 = line.split('\t')
            xs.append(int(x))
            p1s.append(float(p1))
            p2s.append(float(p2))

    fig = plt.figure()
    host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
    par1 = ParasiteAxes(host, sharex=host)
    host.parasites.append(par1)
    host.axis['right'].set_visible(False)
    par1.axis['right'].set_visible(True)
    par1.set_ylabel(par2label)
    par1.axis['right'].major_ticklabels.set_visible(True)
    par1.axis['right'].label.set_visible(True)
    fig.add_axes(host)
    host.set_xlabel(xlabel)
    host.set_ylabel(par1label)
    p1, = host.plot(np.array(xs), np.array(p1s), label=par1label)
    p2, = par1.plot(np.array(xs), np.array(p2s), label=par2label)
    plt.title(title)
    host.legend()
    host.axis['left'].label.set_color(p1.get_color())
    par1.axis['right'].label.set_color(p2.get_color())
    plt.savefig(outimgpath, dpi=150)
    plt.clf()

def drawTwoLoss(chpt_path):
    '''
    :param chpt_path: str, e.g. ./checkpoint/resnet50-CIFAR-10-lr-0.1-dampratio-100-resamplednew/
    '''
    print('=' * 30 + f' now drawing loss curves for {chpt_path} ' + '=' * 30)
    eval_path = os.path.join('./evalres', os.path.basename(os.path.normpath(chpt_path)))
    os.makedirs(os.path.join(eval_path, 'losscurves'), exist_ok=True)

    drawCurveDonkey(os.path.join(chpt_path, 'train_loss.txt'), os.path.join(eval_path, 'losscurves', 'train.png'), title=f"train: {os.path.basename(os.path.normpath(chpt_path))}")
    drawCurveDonkey(os.path.join(chpt_path, 'test_loss.txt'), os.path.join(eval_path, 'losscurves', 'test.png'), title=f"test: {os.path.basename(os.path.normpath(chpt_path))}")

features_net = []
def hook_resnet(module, input, output):
    setattr(module, "_value_hook", output)
    features_net.append(np.squeeze(output.data.cpu().numpy()))
    # features_resnet.append(output.clone().detach())

def load_model(chpt_path):
    print('=' * 30 + f' now loading model for {chpt_path} ' + '=' * 30)
    in_channels = 1 if '-MNIST-' in chpt_path else 3

    datasetname = re.findall(r'MNIST|CIFAR-10|CUB200|Stanford-Dogs', chpt_path)[0]
    netname = re.findall(r'vgg16|resnet50', chpt_path)[0]

    features_names = FEATURE_NAMES[netname]

    num_classes = len(dataloader.SAMPLED_CLS_INDEX[datasetname]) if '-resamplednew' in chpt_path else len(
        dataloader.CLASSES[datasetname])
    net = NETWRKS[netname](num_classes, in_channels)
    checkpoint = torch.load(os.path.join(chpt_path, 'ckpt.pth'), map_location='cpu')
    new_state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['net'].items()}
    net.load_state_dict(new_state_dict)
    net.eval()

    for n, m in net.named_modules():
        if n in features_names:
            m.register_forward_hook(hook_resnet)
    return net, netname, datasetname, features_names


def run_and_draw_featmap(chpt_path, net, testloader, features_names, savedir):
    print('=' * 30 + f' now drawing feature maps for {chpt_path} ' + '=' * 30)
    global features_net
    features_net = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            imgsamplenum = inputs.shape[0]
            outputs = net(inputs).detach()

            for j, batchfeature in enumerate(features_net):
                layername = features_names[j]
                print('*' *20 + f' {layername} ' + '*' * 20)
                for imgind in range(imgsamplenum):
                    label = targets[imgind].item()
                    thisfeature = batchfeature[imgind]
                    channels, size, _ = thisfeature.shape
                    print_chn_num = min(channels, 64)
                    sqcolrow = int(np.ceil(np.sqrt(print_chn_num)))
                    dpi = 256 #min(size ** 2, 256)
                    fig, axarr = plt.subplots(sqcolrow, sqcolrow, figsize=(sqcolrow, sqcolrow), dpi=dpi)
                    for i in range(print_chn_num):
                        row = i // sqcolrow
                        col = i % sqcolrow
                        axarr[(row, col)].imshow(thisfeature[i])

                    fig.suptitle(f'{imgind:02d}-{int(label):02d}.png\'s featuremap after {layername}\nmean={thisfeature.mean()}, var={thisfeature.var()}\nfirst{print_chn_num}/{channels} are shown', fontsize=10)
                    plt.savefig(os.path.join(savedir, f'{imgind:02d}-{int(label):02d}-{layername}.png'))
                    plt.clf()

def run_and_draw_gradcam(chpt_path, net, netname, datasetname, target_index=None):
    print('=' * 30 + f' now drawing Grad-CAM for {chpt_path} ' + '=' * 30)
    image_size = 224 if 'vgg16-' in chpt_path else 32
    eval_path = os.path.join('./evalres', os.path.basename(os.path.normpath(chpt_path)))
    testrawimgsdir = os.path.join(eval_path, 'testrawimgs')
    gradcamsdir = os.path.join(eval_path, 'gradcams')
    os.makedirs(gradcamsdir, exist_ok=True)

    feature_module = FEATURE_MODULES[netname](net)
    target_layer_names = TARGET_LAYER_NAMES[netname]
    grad_cam = gradcam.GradCam(model=net, feature_module=feature_module,
                       target_layer_names=target_layer_names, use_cuda=True if device == 'cuda' else False)
    gb_model = gradcam.GuidedBackpropReLUModel(model=net, use_cuda=True if device == 'cuda' else False)

    for image_name in os.listdir(testrawimgsdir):
        image_path = os.path.join(testrawimgsdir, image_name)
        save_path_base = os.path.join(gradcamsdir, os.path.splitext(image_name)[0])

        img = Image.open(image_path)
        if datasetname != 'MNIST': img = img.convert('RGB')
        input = dataloader.TRANSFORMS(image_size)[datasetname][False](img)
        input.unsqueeze_(0)
        input = input.requires_grad_(True)
        mask = grad_cam(input, target_index)
        cv2img = np.array(img.convert('RGB').resize((image_size, image_size))) #MNIST to 3-channels
        cv2img = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)
        gradcam.show_cam_on_image(cv2img, mask, f'{save_path_base}-cam.jpg')

        gb = gb_model(input, index=target_index)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = gradcam.deprocess_image(cam_mask * gb)
        gb = gradcam.deprocess_image(gb)

        cv2.imwrite(f'{save_path_base}-gb.jpg', gb)
        cv2.imwrite(f'{save_path_base}-cam_gb.jpg', cam_gb)


def testfewforward(chpt_path, net, netname, datasetname, features_names, imgSampleNum=10):
    image_size = 224 if 'vgg16-' in chpt_path else 32
    eval_path = os.path.join('./evalres', os.path.basename(os.path.normpath(chpt_path)))
    os.makedirs(os.path.join(eval_path, 'testrawimgs'), exist_ok=True)
    os.makedirs(os.path.join(eval_path, 'featuremaps'), exist_ok=True)

    testloader = dataloader.loadData(datasetname, image_size=image_size, istrain=False,
                                     resampled=('-resamplednew' in chpt_path),
                                     batch_size=imgSampleNum, isshuffle=False, num_workers=0, ispin_memory=False,
                                     saverawdir=os.path.join(eval_path, 'testrawimgs'), imgSampleNum=imgSampleNum)

    run_and_draw_featmap(chpt_path, net, testloader, features_names, os.path.join(eval_path, 'featuremaps'))


def draw_one_PCA(X, Y, save_path_3D, trydiff=False, var_thresholds=(0.07, 0.05, 0.03)): #trydiff: whether to PCA with different components
    print('=' * 30 + f' now PCA for {save_path_3D} ' + '=' * 30)
    #too slow if using PCA
    try:
        # cannot be used for batchsize
        addtitle = ''
        if trydiff:
            explained_variance_ratio = {}
            for var_thd in var_thresholds:
                pca = IncrementalPCA(n_components=int(var_thd * min(X.shape)), batch_size=int(var_thd * min(X.shape)))
                pca.fit(X)
                explained_variance_ratio[var_thd] = np.sum(pca.explained_variance_ratio_)
            addtitle = ''.join([f'\n{var_thd * 100}%: {int(var_thd * min(X.shape))} components, var_ratio_sum = {explained_variance_ratio[var_thd] * 100:.2f}%' for var_thd in var_thresholds])

        X = (X - np.mean(X)) / np.std(X)
        # assert no nan or inf value in X
        assert np.isnan(X).any() == False
        assert np.isfinite(X).all() == True
        pca = IncrementalPCA(n_components=3, batch_size=10)
        pca.fit(X)
        X_new = pca.transform(X)

        cmap = plt.get_cmap('rainbow', len(Y))
        angles = np.linspace(180, 360, 20)
        i = 0
        for angle in angles:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(10, angle)
            p = ax.scatter3D(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=Y, cmap=cmap, edgecolor='black')
            percent_var_ratio = [f'{x * 100:.2f}%' for x in pca.explained_variance_ratio_]
            fig.suptitle(f'{os.path.basename(os.path.normpath(save_path_3D))}: X in shape {X.shape}\n3rd PCA, var_ratio = {percent_var_ratio}{addtitle}', fontsize=10)
            fig.colorbar(p)
            outfile = os.path.join(save_path_3D, '3dplot_step_' + chr(i + 97) + '.png')
            plt.savefig(outfile, dpi=150)
            plt.clf()
            i += 1
        call(['convert', '-delay', '50', save_path_3D + '/3dplot*', save_path_3D + '/3dplot_anim_' + '.gif'])
    except Exception as ex:
        with open(os.path.join(save_path_3D, 'fail.txt'), 'w') as fout:
            print(ex)
            fout.write(f'{ex}')

def draw_all_PCA(chpt_path):
    PCA_path = os.path.join('./evalres', os.path.basename(os.path.normpath(chpt_path)), 'PCA_all')
    for save_path_3D in [os.path.join(PCA_path, x) for x in os.listdir(PCA_path)]:
        X = np.load(os.path.join(save_path_3D, 'X_tensor.npy'))
        X = X.reshape((X.shape[0], -1))
        Y = np.load(os.path.join(save_path_3D, 'Y.npy'))
        draw_one_PCA(X, Y, save_path_3D)

def testallforward(chpt_path, net, netname, datasetname, features_names):
    image_size, batch_size = (224, 32) if 'vgg16-' in chpt_path else (32, 128)
    eval_path = os.path.join('./evalres', os.path.basename(os.path.normpath(chpt_path)))

    testloader = dataloader.loadData(datasetname, image_size=image_size, istrain=False,
                                     resampled=('-resamplednew' in chpt_path),
                                     batch_size=batch_size, isshuffle=False, num_workers=0, ispin_memory=False)
    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True
    net.eval()

    features_of_all = {features_name: [] for features_name in features_names}
    features_of_all['out'] = []
    targets_of_all = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            global features_net
            features_net = []
            targets_of_all.append(targets.numpy())
            inputs = inputs.to(device)
            outputs = net(inputs).detach()
            outputs = np.squeeze(outputs.data.cpu().numpy())
            for ind, batchfeature in enumerate(features_net):
                features_of_all[features_names[ind]].append(batchfeature)
            features_of_all['out'].append(outputs)

    targets_of_all = np.concatenate(targets_of_all)
    for k, v in features_of_all.items():
        feature_k_of_all = np.concatenate(v)
        save_path_3D = os.path.join(eval_path, 'PCA_all', k)
        os.makedirs(save_path_3D, exist_ok=True)
        np.save(os.path.join(save_path_3D, 'X_tensor.npy'), feature_k_of_all)
        np.save(os.path.join(save_path_3D, 'Y.npy'), targets_of_all)

def drawimggrid(imgs, savedir):
    '''
    :param imgs: a list of PIL Image, in shape of eg. 32x32x3
    '''
    imagegirds = np.empty((CHANNEL_PRINT, NUMS_PER_COL * WIDTH_PRINT, NUMS_PER_COL * WIDTH_PRINT))
    for i in range(len(imgs)):
        row = i // NUMS_PER_COL
        col = i % NUMS_PER_COL
        img = np.rollaxis(np.array(imgs[i].convert('RGB').resize((WIDTH_PRINT, WIDTH_PRINT))), 2, 0) #np, 3x50x50
        imagegirds[:, row * WIDTH_PRINT:row * WIDTH_PRINT + WIDTH_PRINT,
                col * WIDTH_PRINT:col * WIDTH_PRINT + WIDTH_PRINT] = img
    img0 = np.uint8(imagegirds[0])
    img1 = np.uint8(imagegirds[1])
    img2 = np.uint8(imagegirds[2])
    i0 = Image.fromarray(img0)
    i1 = Image.fromarray(img1)
    i2 = Image.fromarray(img2)
    img = Image.merge("RGB", (i0, i1, i2))
    img.save(savedir, "png")

def sort_ftmap(chpt_path, netname, datasetname):
    print('=' * 30 + f' now drawing max activated for {chpt_path} ' + '=' * 30)
    PCA_path = os.path.join('./evalres', os.path.basename(os.path.normpath(chpt_path)), 'PCA_all')
    for save_path_3D in [os.path.join(PCA_path, x) for x in os.listdir(PCA_path)]:
        # X in shape (NUM_OF_IMGS_IN_TESTSET, CHANNEL#, size, size) e.g. (1000, 128, 56, 56)
        ftmaps = np.load(os.path.join(save_path_3D, 'X_tensor.npy'))
        channel_mean = np.mean(ftmaps, axis=(2,3)) if len(ftmaps.shape) == 4 else ftmaps
        layername = os.path.basename(os.path.normpath(save_path_3D))
        max_img_dir = os.path.join('./evalres', os.path.basename(os.path.normpath(chpt_path)), 'max_imgs', layername)
        imagesAll = dataloader.loadDataSetRaw(datasetname, istrain=False, resampled=True)
        os.makedirs(max_img_dir, exist_ok=True)

        for filter_index in PICK_FILTER_INDICES[netname][layername]:
            images_indexs = np.argsort(channel_mean[:,filter_index])
            max_indices = images_indexs[-(NUMS_PER_COL**2):][::-1]
            maximages = [imagesAll[ind][0] for ind in max_indices]
            drawimggrid(maximages, os.path.join(max_img_dir,'%03d_max.png' % filter_index))

            plt.figure()
            plt.hist(channel_mean[:,filter_index], bins=100, normed=False)
            plt.title(f'hist for {layername} {filter_index}-unit output')
            plt.text(-30, -10, 'μ=%.3f, δ=%.3f' % (np.mean(channel_mean[:,filter_index]), np.var(channel_mean[:,filter_index])))
            plt.savefig(os.path.join(max_img_dir,'%03d_hist.png' % filter_index))
            plt.clf()


if __name__ == '__main__':
    chpt_bases = ['resnet50-CIFAR-10-lr-0.1-dampratio-100-resamplednew', 'resnet50-CUB200-lr-0.05-dampratio-100-resamplednew',
                  'resnet50-MNIST-lr-0.1-dampratio-100-resamplednew', 'resnet50-Stanford-Dogs-lr-0.05-dampratio-100-resamplednew',
                  'vgg16-CIFAR-10-lr-0.01-dampratio-100-resamplednew',  'vgg16-CUB200-lr-0.01-dampratio-100-resamplednew-300epochs',
                  'vgg16-MNIST-lr-0.01-dampratio-100-resamplednew', 'vgg16-Stanford-Dogs-lr-0.01-dampratio-100-resamplednew-300epochs']


    for chpt_base in chpt_bases:
        print('\n' + '= - ' * 30 + '\n')
        
        chpt_path = os.path.join('./checkpoint/', chpt_base)
        net, netname, datasetname, features_names = load_model(chpt_path)
        
        drawTwoLoss(chpt_path)
        
        testfewforward(chpt_path, net, netname, datasetname, features_names, imgSampleNum=5)
        run_and_draw_gradcam(chpt_path, net, netname, datasetname, target_index=None)

        testallforward(chpt_path, net, netname, datasetname, features_names)
        sort_ftmap(chpt_path, netname, datasetname)
        draw_all_PCA(chpt_path)
