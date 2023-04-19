import numpy as np
import torch
from .base_model import BaseModel
from data.image_ct_dataset import ImageCTDataset
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import torch
import numpy as np
from sklearn.manifold._t_sne import TSNE
import torch

import contextual_loss as cl
import contextual_loss.functional as F


from scipy.misc import face
import matplotlib.pyplot as plt

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance', type=util.str2bool, nargs='?', const=True, default=False, help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.add_argument('--lambda_contextual', type=float, default=1.0, help='weight for contextual loss')
        parser.add_argument('--contextual_loss', type=util.str2bool, nargs='?', const=True, default=False, help='if contextual loss will be added to training.')
        
        parser.set_defaults(pool_size=0, paired=False)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G']
        if opt.lambda_NCE > 0:
            self.loss_names.append('NCE')
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            if opt.lambda_NCE > 0:
                self.loss_names.append('NCE_Y')
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.networks = [self.netG]
        if self.isTrain:
            self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.networks.extend([self.netD, self.netF])
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, dtype=torch.float16 if opt.amp else torch.float32).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            self.set_input(data)
            bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
            self.real_A = self.real_A[:bs_per_gpu]
            self.real_B = self.real_B[:bs_per_gpu]
            with torch.no_grad():
                self.forward()                     # compute fake images: G(A)
                if self.opt.isTrain:
                    self.compute_D_loss()                # calculate gradients for D
                    self.compute_G_loss()                   # calculate gradients for G
                    if self.opt.lambda_NCE > 0.0:
                        self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                        self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        # Casts operations to mixed precision
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            self.forward()
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
        self.scaler.scale(self.loss_D).backward()
        self.scaler.step(self.optimizer_D)
        # Updates the scale for next iteration
        torch.cuda.empty_cache()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE>0:
            self.optimizer_F.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.opt.amp):
            self.loss_G = self.compute_G_loss()
        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer_G)
        # Updates the scale for next iteration
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE>0:
            self.scaler.step(self.optimizer_F)
            # Updates the scale for next iteration
        self.scaler.update()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        real_A_in = self.real_A
        if self.opt.nce_idt and self.opt.isTrain:
            real_B_in = self.real_B.expand([*self.real_A.shape[:2], *self.real_B.shape[2:]])
        if self.opt.lambda_NCE and self.opt.phase == 'train':
            self.fake_B, self.real_A_feats = self.netG(real_A_in, self.nce_layers)
        else:
            self.fake_B = self.netG(real_A_in)
        if self.registration_artifacts_idx is not None:
            self.fake_B = self.fake_B * self.registration_artifacts_idx.to(self.fake_B.device)
        if self.opt.lambda_NCE and self.opt.nce_idt and self.opt.isTrain:
            self.idt_B, self.real_B_feats = self.netG(real_B_in, self.nce_layers)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A_feats, self.fake_B.expand(self.real_A.shape))
            self.real_A_feats = None
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B_feats, self.idt_B.expand([*self.real_A.shape[:2], *self.real_B.shape[2:]]))
            self.real_B_feats = None
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        if self.opt.contextual_loss and self.opt.lambda_contextual > 0.0:
            print(np.shape(self.real_A))
            # print(np.shape(self.fake_B))
            print(np.shape(self.real_B))
            # print(ImageCTDataset.A_paths)
            # print(ImageCTDataset.B_paths)
            # plt.imshow(self.fake_B[2:])
            # plt.imshow(self.real_B[2:])
            self.contextual_loss = F.contextual_loss(self.fake_B, self.real_B, band_width=0.1, loss_type='cosine')
        else:
            self.contextual_loss = 0.0

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.contextual_loss
        return self.loss_G

    def calculate_NCE_loss(self, feat_k, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        # feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def log_gradients(model):
            all_grads = {}
            all_weights = {}
            for n, p in model.named_parameters():
                if p.grad is not None:
                    all_grads[n] = np.mean(p.grad.cpu().numpy())
                    all_weights[n] = p 
            return all_grads, all_weights

    def compute_CX_loss(img1, img2):
        # contextual loss
        criterion = cl.ContextualLoss()
        loss = criterion(img1, img2)

        loss = F.contextual_loss(img1, img2, band_width=0.1, loss_type='cosine')
        return loss




######## Below this line is the implementation of the Contextual Loss ########

# import tensorflow as tf
class TensorAxis:
    N = 0
    H = 1
    W = 2
    C = 3


class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=TensorAxis.C):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = torch.exp((self.b - scaled_distances) / self.sigma)
        # self.cs_weights_before_normalization = 1 / (1 + scaled_distances)
        # self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)
        self.cs_NHWC = self.cs_weights_before_normalization

    # def reversed_direction_CS(self):
    #     cs_flow_opposite = CSFlow(self.sigma, self.b)
    #     cs_flow_opposite.raw_distances = self.raw_distances
    #     work_axis = [TensorAxis.H, TensorAxis.W]
    #     relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
    #     cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
    #     return cs_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        r_Ts = torch.sum(Tvecs * Tvecs, 2)
        r_Is = torch.sum(Ivecs * Ivecs, 2)
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
            A = Tvec @ torch.transpose(Ivec, 0, 1)  # (matrix multiplication)
            cs_flow.A = A
            # A = tf.matmul(Tvec, tf.transpose(Ivec))
            r_T = torch.reshape(r_T, [-1, 1])  # turn to column vector
            dist = r_T - 2 * A + r_I
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_L1(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec = Ivecs[i], Tvecs[i]
            dist = torch.abs(torch.sum(Ivec.unsqueeze(1) - Tvec.unsqueeze(0), dim=2))
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        # prepare feature before calculating cosine distance
        T_features, I_features = cs_flow.center_by_T(T_features, I_features)
        T_features = CSFlow.l2_normalize_channelwise(T_features)
        I_features = CSFlow.l2_normalize_channelwise(I_features)

        # work seperatly for each example in dim 1
        cosine_dist_l = []
        N = T_features.size()[0]
        for i in range(N):
            T_features_i = T_features[i, :, :, :].unsqueeze_(0)  # 1HWC --> 1CHW
            I_features_i = I_features[i, :, :, :].unsqueeze_(0).permute((0, 3, 1, 2))
            patches_PC11_i = cs_flow.patch_decomposition(T_features_i)  # 1HWC --> PC11, with P=H*W
            cosine_dist_i = torch.nn.functional.conv2d(I_features_i, patches_PC11_i)
            cosine_dist_1HWC = cosine_dist_i.permute((0, 2, 3, 1))
            cosine_dist_l.append(cosine_dist_i.permute((0, 2, 3, 1)))  # back to 1HWC

        cs_flow.cosine_dist = torch.cat(cosine_dist_l, dim=0)

        cs_flow.raw_distances = - (cs_flow.cosine_dist - 1) / 2  ### why -

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    def calc_relative_distances(self, axis=TensorAxis.C):
        epsilon = 1e-5
        div = torch.min(self.raw_distances, dim=axis, keepdim=True)[0]
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    @staticmethod
    def sum_normalize(cs, axis=TensorAxis.C):
        reduce_sum = torch.sum(cs, dim=axis, keepdim=True)
        cs_normalize = torch.div(cs, reduce_sum)
        return cs_normalize

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size
        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT = T_features.mean(0, keepdim=True).mean(1, keepdim=True).mean(2, keepdim=True)
        self.varT = T_features.var(0, keepdim=True).var(1, keepdim=True).var(2, keepdim=True)
        self.T_features_centered = T_features - self.meanT
        self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = features.norm(p=2, dim=TensorAxis.C, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, T_features):
        # 1HWC --> 11PC --> PC11, with P=H*W
        (N, H, W, C) = T_features.shape
        P = H * W
        patches_PC11 = T_features.reshape(shape=(1, 1, P, C)).permute(dims=(2, 3, 0, 1))
        return patches_PC11

    @staticmethod
    def pdist2(x, keepdim=False):
        sx = x.shape
        x = x.reshape(shape=(sx[0], sx[1] * sx[2], sx[3]))
        differences = x.unsqueeze(2) - x.unsqueeze(1)
        distances = torch.sum(differences**2, -1)
        if keepdim:
            distances = distances.reshape(shape=(sx[0], sx[1], sx[2], sx[3]))
        return distances

    @staticmethod
    def calcR_static(sT, order='C', deformation_sigma=0.05):
        # oreder can be C or F (matlab order)
        pixel_count = sT[0] * sT[1]

        rangeRows = range(0, sT[1])
        rangeCols = range(0, sT[0])
        Js, Is = np.meshgrid(rangeRows, rangeCols)
        row_diff_from_first_row = Is
        col_diff_from_first_col = Js

        row_diff_from_first_row_3d_repeat = np.repeat(row_diff_from_first_row[:, :, np.newaxis], pixel_count, axis=2)
        col_diff_from_first_col_3d_repeat = np.repeat(col_diff_from_first_col[:, :, np.newaxis], pixel_count, axis=2)

        rowDiffs = -row_diff_from_first_row_3d_repeat + row_diff_from_first_row.flatten(order).reshape(1, 1, -1)
        colDiffs = -col_diff_from_first_col_3d_repeat + col_diff_from_first_col.flatten(order).reshape(1, 1, -1)
        R = rowDiffs ** 2 + colDiffs ** 2
        R = R.astype(np.float32)
        R = np.exp(-(R) / (2 * deformation_sigma ** 2))
        return R






# --------------------------------------------------
#           CX loss
# --------------------------------------------------



def calculate_CX_loss(T_features, I_features, deformation=False, dis=False):
    # T_features = tf.convert_to_tensor(T_features, dtype=tf.float32)
    # I_features = tf.convert_to_tensor(I_features, dtype=tf.float32)
    # since this is a convertion of tensorflow to pytorch we permute the tensor from
    # T_features = normalize_tensor(T_features)
    # I_features = normalize_tensor(I_features)

    # since this originally Tensorflow implemntation
    # we modify all tensors to be as TF convention and not as the convention of pytorch.
    def from_pt2tf(Tpt):
        Ttf = Tpt.permute(0, 2, 3, 1)
        return Ttf
    # N x C x H x W --> N x H x W x C
    T_features_tf = from_pt2tf(T_features)
    I_features_tf = from_pt2tf(I_features)

    # cs_flow = CSFlow.create_using_dotP(I_features_tf, T_features_tf, sigma=1.0)
    cs_flow = CSFlow.create_using_L2(I_features_tf, T_features_tf, sigma=1.0)
    # sum_normalize:
    # To:
    cs = cs_flow.cs_NHWC

    if deformation:
        deforma_sigma = 0.001
        sT = T_features_tf.shape[1:2 + 1]
        R = CSFlow.calcR_static(sT, deformation_sigma=deforma_sigma)
        cs *= torch.Tensor(R).unsqueeze(dim=0).cuda()

    if dis:
        CS = []
        k_max_NC = torch.max(torch.max(cs, dim=1)[1], dim=1)[1]
        indices = k_max_NC.cpu()
        N, C = indices.shape
        for i in range(N):
            CS.append((C - len(torch.unique(indices[i, :]))) / C)
        score = torch.FloatTensor(CS)
    else:
        # reduce_max X and Y dims
        # cs = CSFlow.pdist2(cs,keepdim=True)
        k_max_NC = torch.max(torch.max(cs, dim=1)[0], dim=1)[0]
        # reduce mean over C dim
        CS = torch.mean(k_max_NC, dim=1)
        # score = 1/CS
        # score = torch.exp(-CS*10)
        score = -torch.log(CS)
    # reduce mean over N dim
    # CX_loss = torch.mean(CX_loss)
    return score


def symetric_CX_loss(T_features, I_features):
    score = (calculate_CX_loss(T_features, I_features) + calculate_CX_loss(I_features, T_features)) / 2
    return score