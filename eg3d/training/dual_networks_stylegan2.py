# 这个代码除了基础的dynagan之外，另外还要加入一个mapping网络来扩大影响力

"""Network architectures from the paper
"Analyzing and Improving the Image Quality of StyleGAN".
Matches the original implementation of configs E-F by Karras et al. at
https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py"""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
import math
import snoop
from torch import nn

class AdaptiveInstanceNorm(nn.Module):
    # 这个就adain
    def __init__(self, fin, style_dim=512):
        super().__init__()

        self.norm = nn.InstanceNorm2d(fin, affine=False)
        self.style = nn.Linear(style_dim, fin * 2).cuda()

        self.style.bias.data[:fin] = 1
        self.style.bias.data[fin:] = 0

    # @snoop()
    def forward(self, input, style):
        # style.shape = (1, 512)
        # input.shape = (1, 512, 4, 4)
        style = self.style(style).unsqueeze(2).unsqueeze(3) # style.shape = (1, 192, 1, 1)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta # gamma.shape = (1, 96, 1, 1) out.shape = (1, 512, 4, 4) beta.shape = (1, 96, 1, 1)
        return out
    
class AdaResBlock(nn.Module):
    def __init__(self, fin, style_dim=512):
        super().__init__()

        self.conv = Conv2dLayer(fin, fin, kernel_size=3)
        self.conv2 = Conv2dLayer(fin, fin, kernel_size=3)

        self.norm = AdaptiveInstanceNorm(fin, style_dim)
        self.norm2 = AdaptiveInstanceNorm(fin, style_dim)
        
        # model initialization
        # the convolution filters are set to values close to 0 to produce negligible residual features
        self.conv.weight.data *= 0.01
        self.conv2.weight.data *= 0.01
        
    # @snoop()
    def forward(self, x, s, w=1):
        # resnet的结构
        skip = x
        if w == 0:
            return skip
        out = self.conv(self.norm(x, s))
        out = self.conv2(self.norm2(out, s))
        out = out * w + skip
        return out
#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------
@misc.profiled_function
def modulated_conv2d_origin(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


# 这个是核心调制模块，如果要加入DynaGAN成分，那应该也是修改这里
# 出现于：SynthesisLayer和ToRGBLayer
# @snoop()
@misc.profiled_function
def modulated_conv2d( 
    adaResBlock,hypernet,str,style_dim,out_channels, in_channels,kernel_size, 
    x,  #  (1, 128, 512, 512)   # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight, # weight.shape = (128, 128, 3, 3) 或者weight.shape = (3, 128, 1, 1)  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,  # styles.shape = (1, 128)  # Modulation coefficients of shape [batch_size, in_channels]
    
    domain_style=None,alpha=1.0, beta=1.0,
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0] # batch_size = 1
    out_channels, in_channels, kh, kw = weight.shape # (128,128,3,3) 或者(3,128,1,1)
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]
    
    
    # Pre-normalize inputs to avoid FP16 overflow.调制输入的style和weight
    if x.dtype == torch.float16 and demodulate: # 运行了
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    """print(f"使用modulate的层为：{str}层")
    print("开始hypernet的forward")
    print("hypernet的x输入.shape",x.shape)"""
    # ---------------------------------核心修改开始-------------------------------------- #
    # 申明超网络,成功
    no_scaling=False
    no_residual=False
    hypernet_now=hypernet
    # hypernet = create_Hypernet(style_dim, out_channels, in_channels,kernel_size,no_scaling=no_scaling, no_residual=no_residual)

    # 这里申明一下adaresblock
    # res_block=adaResBlock

    #使用超网络出结果，成功
    # print("domain_style",domain_style)
    # print("domain_style[0].shape",domain_style[0].shape) # torch.Size([1, 512])
    if domain_style is not None: # 确定了，这个domain_style就是(n,512)
        """print("运行了hypernet的层")"""
        res_weight, domain_mod = hypernet_now(domain_style[0].to('cuda'),x)  # 超网络，映射得到 res_weight, domain_mod
        # print("res_weight.shape",res_weight.shape)
        # print("domain_mod.shape",domain_mod.shape)
        res_weight = beta * res_weight
        domain_mod = alpha*(domain_mod-1.0)+1.0 
    else:
        res_weight, domain_mod = 0.0, 1.0 # 初始化

    # 新加入的层级融合
    # if y!=0: # 代码有问题，没有真正的融合
    #   
    #     x=x*0.5+y*0.5
    
    # 打印记录在日志里面
    # with open('/data1/dxw_data/code/SPI/testlog/metric_log.txt', "a") as log_file: # 累计记录，不会清空上一次
    #     message = f'res_weight: {res_weight}\n'
    #     message += f'domain_mod: {domain_mod}\n'
    #     log_file.write(message)
    #     log_file.write('\n')


    # Calculate per-sample weights and demodulation coefficients.得到w
    fan_in = in_channels * kernel_size ** 2
    scale = 1 / math.sqrt(fan_in)
    w = None
    dcoefs = None
    if demodulate or fused_modconv:  # 运行了
        """print("@@@@@@@@@")"""
        # 原始结果
        # w = weight.unsqueeze(0) # [NOIkk] w.shape = (1, 128, 128, 3, 3) 或者 w.shape = (1, 3, 128, 1, 1)
        # w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk] 其实style本身就是一个A的MLP了。
        # 关键修改的地方，确定正确，除了scale不知道是否正确，这里可能有问题
        # 新的结果：
        w = weight.unsqueeze(0) # [NOIkk]
        w =(w+res_weight) * styles.reshape(batch_size, 1, -1, 1, 1) 
        # 绝对不要使用scale
        # w =scale* (w+res_weight) * styles.reshape(batch_size, 1, -1, 1, 1) 

    """
    测试后的结论：
    上述的例子可以运行，可以正常通过（前提是不能有scale，否则就会完全失败，绝对不能使用scale）
    原始方法和我的hypernet的方法都可以运行，都可以出人脸
    具体如何调参，使用几个分支，如何讲故事还不是我说了算
    """

    # 这个模块的意义也没有搞清楚，感觉可能有问题
    if demodulate:  # 前面阶段会运行，后面阶段不会运行 
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO] coefs.shape = (1, 128) coefs其实就是图里面的demod部分   dcoefs.shape = (25, 512)，w.shape = (25, 512, 512, 3, 3)
    if demodulate and fused_modconv: # 前面阶段会运行，后面阶段不会运行 
        # 原始：
        # w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]
        # 新的结果：
        w = (w+res_weight)  * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]
        # 绝对不要使用scale
        # w = scale* (w+res_weight)  * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]
    # Execute by scaling the activations before and after the convolution.得到x
    if not fused_modconv: # 这一整个段落都没有运行
        # print("###########") # 没有运行过
        # assert 0 # 实验！！如果有运行这里就直接退出，看起来是完全不运行的
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        # print("最终的x的格式: x.shape",x.shape)
        return x

    # Execute as one fused op using grouped convolution. 开始混合w和x，下面一整个段落都运行了
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    # 原始：
    # 无
    # 新的结果
    w=w  * domain_mod # 确定了格式没有问题，变化前后都是torch.Size([1, 512, 512, 3, 3])
    # print("w.shape",w.shape)
    w = w.reshape(-1, in_channels, kh, kw) # w.shape = (12800, 512, 3, 3)
    # 在这一步结合w和x,这里就是upsample和downsample的部分了。
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight) # x.shape = (1, 3, 512, 512)  x.shape = (1, 12800, 4, 4)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    # ---------------------- # 
    # 经过调查，在这里实现一个conv的out的cat就可以实现dualstylegan的结果，真正的缝合的地方就在这里哈。锁定了位置

    # 能够和x相加的只有x
    # ---------------------- #
    if noise is not None:
        
        # print("noise.shape",noise.shape)
        # print("x.shape",x.shape) # 实验直接在这里添加
        """
        noise.shape torch.Size([128, 128])
        x.shape torch.Size([1, 256, 128, 128])
        noise.shape torch.Size([256, 256])
        x.shape torch.Size([1, 128, 256, 256])
        noise.shape torch.Size([32, 32])
        x.shape torch.Size([1, 512, 32, 32])
        """
    
        x = x.add_(noise)
  
    # ---------------------------------核心修改结束-------------------------------------- #
    # if x.shape[1]==512:
        """print("&&&&&&&&&&&&&&&&&&&-开始")
        print("x.shape",x.shape)"""
        # y=res_block(x, domain_style[0].to('cuda'))
        """print("y.shape",y.shape)
        print("进行了融合")"""
        # x=x+0.75*y # 0.75的权重，已经这种连接方法，是从dualstylegan里面学到的，不是空穴来风
        """print("融合后的x的shape",x.shape)
        print("&&&&&&&&&&&&&&&&&&&-结束")"""
    """print("最终的x的格式:：x.shape",x.shape)
    print("------------------------------------------------------------------------------------------")"""
    return x

def create_Hypernet(style_dim, out_channels, in_channels,kernel_size,no_scaling=False, no_residual=False):
    # 返回初始化的结果
    return HyperNet(style_dim, out_channels, in_channels,kernel_size,no_scaling, no_residual)

def create_AdaResBlock(style_dim):
    return AdaResBlock(fin=style_dim)

#----------------------------------------------------------------------------
# 这个要做hpyernetwork的复制
@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    # @snoop()
    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            # print("b",b)
            # print("w",w)
            # x.shape = (1, 1) b.shape = (512,) w.shape = (512, 25)
            # b.unsqueeze(0)=(1,512)
            x = torch.addmm(b.unsqueeze(0), x, w.t()) # 这里就相当于进入了全连接层了
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------
# 这个要做hpyernetwork的复制
@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        # self.lr_multiplier=lr_multiplier

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer) # setattr() 函数对应函数 getattr()，用于设置属性值，该属性不一定是存在的。setattr(x, 'y', v) is equivalent to ``x.y = v''

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    # @snoop() # 似乎就没有人运行过这个
    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}') # getattr(x, 'y') is equivalent to x.y.
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------
# 自己写个模块
@persistence.persistent_class
class HyperNet(torch.nn.Module):
    def __init__(
        self,
        style_dim,
        out_channel,
        in_channel,
        kernel_size,
        no_scaling=False, 
        no_residual=False,
        device = "cuda"
    ):
        super().__init__()
        self.style_dim = style_dim
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        
        self.no_scaling = no_scaling
        self.no_residual= no_residual
        
        # ------------------------------------加载额外的预训练参数的秘密------------------------------------ #
        # 概括就是两个mapping都使用EqualLinear，但是这个提出的mapping的权重调整的很小，所以基本可以直接使用之前的EqualLinear的预训练权重
        if not no_residual: 
            # 定义网络，是正确的初始化，结构是几乎完全一致
            self.out_channel_estimator = FullyConnectedLayer(self.style_dim, self.out_channel, bias_init=1).to(device)
            self.in_channel_estimator = FullyConnectedLayer(self.style_dim, self.in_channel, bias_init=1).to(device)
            self.kernel_estimator = FullyConnectedLayer(self.style_dim, self.kernel_size ** 2, bias_init=0).to(device)
        
            # weight and  bias init, which are negligible to the main generator：weight 和 bias init，对主生成器来说可以忽略不计,直接可以提取data，然后使用权重
            # 定义权重，自己对于自
            # 原始的权重是0.01
            self.out_channel_estimator.weight.data = self.out_channel_estimator.weight.data  * 0.01
            self.in_channel_estimator.weight.data = self.in_channel_estimator.weight.data  * 0.01
            self.kernel_estimator.weight.data = self.kernel_estimator.weight.data  * 0.01
            
        if not no_scaling:
            # 定义网络
            self.domain_modulation = FullyConnectedLayer(self.style_dim, self.out_channel, bias_init=1).to(device)
            # 定义权重
            self.domain_modulation.weight.data = self.domain_modulation.weight.data * 0.01
        # print("初始化hypernet成功")

    # @snoop()
    def forward(self, domain_style,x):
        batch = len(domain_style) # batch=25
        if not self.no_residual:
            res_out_channel = self.out_channel_estimator(domain_style) # res_out_channel.shape = (25, 64)
            res_in_channel = self.in_channel_estimator(domain_style) # res_in_channel.shape = (25, 64)
            res_kernel = self.kernel_estimator(domain_style) # res_kernel.shape = (25, 9)
            res_conv = torch.bmm(res_out_channel.unsqueeze(-1), torch.bmm( res_in_channel.unsqueeze(-1), res_kernel.unsqueeze(1)).view(batch, -1).unsqueeze(1)).view(batch, self.out_channel, self.in_channel, self.kernel_size, self.kernel_size) # res_conv.shape = (25, 64, 64, 3, 3)
        else:
            res_conv = 0. # 初始化
        
        if not self.no_scaling: # 这里是全部都会运行的！！
            # 负责将全连接层，输出结果，这个就是A吧
            domain_mod = self.domain_modulation(domain_style).view(batch, self.out_channel, 1, 1, 1) # domain_mod.shape = (25, 64, 1, 1, 1)或domain_mod.shape = (1, 256, 1, 1, 1)或domain_mod.shape = (1, 96, 1, 1, 1)或domain_mod.shape = (1, 512, 1, 1, 1)
        else:
            domain_mod = 1. # 初始化
        # print("运行hypernet成功")

        
        return res_conv, domain_mod

#----------------------------------------------------------------------------
# 这个模块基本不动
@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    # @snoop()
    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to('cuda') if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        # x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)
        x = conv2d_resample.conv2d_resample(x=x, w=w.to('cuda'), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        # 这里就是加入的残差
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
            f'up={self.up}, down={self.down}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        # 测试新加入的
        self.kernel_size = kernel_size

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        
    # @snoop()
    def forward(self,domain_style, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        # 这里生成noise，但是没有bias
        # 优点在于可以很好的匹配格式，而且就是一个随机的向量

        flip_weight = (self.up == 1) # slightly faster
        # print("self.bias.shape",self.bias.shape)
        # print("x.shape",x.shape)
        """
        # 毕竟：self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        # 无法对齐？？
        self.bias.shape torch.Size([512])
        x.shape torch.Size([1, 512, 32, 32])
        self.bias.shape torch.Size([512])
        x.shape torch.Size([1, 512, 64, 64])
        self.bias.shape torch.Size([256])
        x.shape torch.Size([1, 512, 64, 64])
        self.bias.shape torch.Size([256])
        x.shape torch.Size([1, 256, 128, 128])
        """
        x = modulated_conv2d(adaResBlock=self.adaResBlock,hypernet=self.hypernet,str='SynthesisLayer',style_dim= self.w_dim,out_channels=self.out_channels, in_channels=self.in_channels ,kernel_size=self.kernel_size,domain_style=domain_style, x=x, weight=self.weight, styles=styles, noise=noise, up=self.up, padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)
        
        # ---------------------- # 
        # 经过调查，在这里实现一个conv的out的cat就可以实现dualstylegan的结果，真正的缝合的地方就在这里哈。锁定了位置

        # 决定了还是不写这里
        # ---------------------- #

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        
        # # 测试使用hypernetwork：成功！！！
        # no_scaling=False
        # no_residual=False
        # hypernet = HyperNet(self.w_dim, self.out_channels, self.in_channels,self.kernel_size,no_scaling=no_scaling, no_residual=no_residual)
        # domain_style=torch.rand([25,512]).to('cuda')
        # res_weight, domain_mod = hypernet(domain_style)
        
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])
    
    def create_domain_modulation(self, no_scaling=False, no_residual=False):
        self.hypernet=create_Hypernet(style_dim= self.w_dim,out_channels=self.out_channels, in_channels=self.in_channels ,kernel_size=self.kernel_size,no_scaling=no_scaling, no_residual=no_residual)
        # print("成功初始化hypernet")
        self.adaResBlock=create_AdaResBlock(style_dim= self.w_dim)

#----------------------------------------------------------------------------
# 
@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        # 新加的：
        self.kernel_size=kernel_size


    def forward(self,domain_style, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d_origin(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

#----------------------------------------------------------------------------
# 这个模块基本不动
@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):

    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        # 新加的：
        self.out_channels=out_channels
        self.kernel_size = 3

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, domain_style, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(domain_style, x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(domain_style,x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(domain_style,x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(domain_style,x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(domain_style,x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(domain_style, x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

    def create_domain_modulation(self, no_scaling=False, no_residual=False):
        # 需要导入一个变量来控制次数。
        if self.in_channels!=0:
            # print("申明conv0")
            self.conv0.create_domain_modulation(no_scaling=False, no_residual=False)
        # print("申明conv1")
        self.conv1.create_domain_modulation(no_scaling=False, no_residual=False)
        # 在这里可能有点问题

#----------------------------------------------------------------------------
# 这个模块基本不动
@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        

        self.num_ws = 0
        # self.blocks= torch.nn.ModuleList()
        for res in self.block_resolutions:
            # print("测试@@")
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            # # 这里需要加入一个新的东西:但是可能会覆盖的
            # self.block.append(block)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block) # self.res的值为block

    # @snoop()
    def forward(self,domain_style, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim]) # 这个感觉像是无法格式匹配的问题了。
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions: # 这里是为了获得w
                block = getattr(self, f'b{res}') # 每个res就是一个block
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb)) # 这里就逐渐保存为（1,14,512）
                w_idx += block.num_conv
        x = img = None
        # print("block_ws",block_ws)
        for res, cur_ws in zip(self.block_resolutions, block_ws): # 对于每一层的w和res-block而言，开始正常的产生结果
            block = getattr(self, f'b{res}')
            x, img = block(domain_style, x, img, cur_ws, **block_kwargs)
            # print("x.shape",x.shape)
            # print("img.shape",img.shape)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

    def create_domain_modulation(self, no_scaling=False, no_residual=False):

        # i=0 # 可以通过这个函数来控制层的数量和从粗到细吗，结论是可以控制
        # 这里就是可以控制次数的地方
        # i=0
        # print("self.block_resolutions",self.block_resolutions) # [4, 8, 16, 32, 64, 128, 256]
        for res in self.block_resolutions:
            # print("i:",i)
            block = getattr(self, f'b{res}')
            block.create_domain_modulation(no_scaling=no_scaling, no_residual=no_residual)
            # i=i+1
            # if i>5:
            #     block = getattr(self, f'b{res}')
            #     block.create_domain_modulation(no_scaling=no_scaling, no_residual=no_residual)
            # else:
            #     pass

#----------------------------------------------------------------------------
# 这个模块基本不动
@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        multi_domain,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.               
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.multi_domain = multi_domain
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        # self.deformstyle = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs) # 加一句这个可以直接提高和修改网络的结构
        self.embedding = None

    # @snoop() # 只有在初始化的load_util中才会什么一次，而且声明的参数来自于保存的pkl，如self.c_dim等=25值是固定的哈
    def create_domain_modulation(self, no_scaling=False, no_residual=False):
        self.synthesis.create_domain_modulation(no_scaling=no_scaling, no_residual=no_residual)
        # print("self.c_dim",self.c_dim)
        # print("self.w_dim",self.w_dim)

        layers = [FullyConnectedLayer(self.c_dim, self.w_dim)]
        for idx in range(2):
            layers.append(
                FullyConnectedLayer(self.w_dim, self.w_dim, lr_multiplier=0.01, activation="lrelu")
            )
        self.embedding = nn.Sequential(*layers)# 以模块作为元素的神经网络容器\
        # print("self.embedding",self.embedding)
        self.is_domain_modulation = True

        # self.embedding Sequential(
        # (0): FullyConnectedLayer(in_features=25, out_features=512, activation=linear)
        # (1): FullyConnectedLayer(in_features=512, out_features=512, activation=fused_lrelu)
        # (2): FullyConnectedLayer(in_features=512, out_features=512, activation=fused_lrelu)
        # )
    # @snoop()
    def get_domain_style(self,domain_is_latents,domain_labels):
        if domain_labels!=[None]:
            # print("###################################")
            # print("domain_labels",domain_labels)
            assert self.embedding is not None  # self.embedding 就是全连接层
            if domain_is_latents:
                domain_styles = domain_labels 
            else: # 基本的都是运行下面
                domain_styles = [self.embedding(l) for l in domain_labels] # 这是是dynaGAN的新分支得到的结果。这几乎
                # print("domain_styles.shape",domain_styles.shape)
        else:
            domain_styles = [None] # # 实际运行
        return domain_styles

    def forward(self,domain_is_latents,domain_labels,z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # 测试，这里切断和外面的domain_style联系，仅仅用于测试
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        # z = self.deformstyle(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

        if self.is_domain_modulation and self.multi_domain and not None in domain_labels:
            # print("###################################")
            # print("domain_labels",domain_labels)
            assert self.embedding is not None  # self.embedding 就是全连接层
            if domain_is_latents:
                domain_styles = domain_labels 
            else: # 基本的都是运行下面
                domain_styles = [self.embedding(l) for l in domain_labels] # 这是是dynaGAN的新分支得到的结果。这几乎
        else:
            domain_styles = [None] # # 实际运行
        # print("domain_styles[0].shape",domain_styles[0].shape) # 检查一下是否为(1,512)

        img1 = self.synthesis(domain_styles, ws, update_emas=update_emas, **synthesis_kwargs)
        # img2 = self.synthesis(domain_style, z, update_emas=update_emas, **synthesis_kwargs)
        return img1#+img2

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------