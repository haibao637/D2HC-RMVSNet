#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
MVSNet sub-models.
"""

from cnn_wrapper.network import Network
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
import math
########################################################################################
############################# 2D feature extraction nework #############################
########################################################################################

class UniNetDS2(Network):
    """Simple UniNet, as described in the paper."""

    def setup(self):
        print ('2D with 32 filters')
        base_filter = 16
        (self.feed('data')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='conv0_0')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='conv0_1')
        .conv_bn(5, base_filter * 2, 2, center=True, scale=True, name='conv1_0')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_1')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_2')
        .conv_bn(5, base_filter * 4, 2, center=True, scale=True, name='conv2_0')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='conv2_1')
        .conv(3, base_filter * 4, 1, biased=False, relu=False, name='conv2_2'))

class UniNetDS2GN(Network):
    """Simple UniNet with group normalization."""

    def setup(self):
        print ('2D with 32 filters')
        base_filter = 32
        (self.feed('data')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='conv0_0')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='conv0_1')
        .conv_gn(5, base_filter * 2, 2, center=True, scale=True, name='conv1_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_1')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_2')
        .conv_gn(5, base_filter * 4, 2, center=True, scale=True, name='conv2_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='conv2_1')
        .conv(3, base_filter * 4, 1, biased=False, relu=False, name='conv2_2'))
class UNetDS2GN0(Network):
    """2D U-Net with group normalization."""

    def setup(self):
        print ('2D UNet with 32 channel output')
        base_filter = 8
        (self.feed('data')
        .conv_gn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv1_0')
        .conv_gn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv2_0')
        .conv_gn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv3_0')
        .conv_gn(3, base_filter * 16, 2, center=True, scale=True, name='2dconv4_0'))

        (self.feed('data')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv0_1')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv0_2'))

        (self.feed('2dconv1_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_1')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_2'))

        (self.feed('2dconv2_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_1')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_2'))

        (self.feed('2dconv3_0')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_1')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_2'))

        (self.feed('2dconv4_0')
        .conv_gn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_1')
        .conv_gn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_2')
        .deconv_gn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv5_0'))

        (self.feed('2dconv5_0', '2dconv3_2')
        .concat(axis=-1, name='2dconcat5_0')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_1')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_2')
        .deconv_gn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv6_0'))

        (self.feed('2dconv6_0', '2dconv2_2')
        .concat(axis=-1, name='2dconcat6_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_1')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_2')
        .deconv_gn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv7_0'))

        (self.feed('2dconv7_0', '2dconv1_2')
        .concat(axis=-1, name='2dconcat7_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_1')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_2')
        .deconv_gn(3, base_filter, 2, center=True, scale=True, name='2dconv8_0'))

        (self.feed('2dconv8_0', '2dconv0_2')
        .concat(axis=-1, name='2dconcat8_0')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv8_1')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv8_2')   # end of UNet
        .conv_gn(5, base_filter * 2, 2, center=True, scale=True, name='conv9_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='conv9_1')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='conv9_2')
        .conv_gn(5, base_filter * 4, 2, center=True, scale=True, name='conv10_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='conv10_1')
        .conv(3, base_filter * 4, 1,biased=False, relu = False, name='conv10_2'))
class UNetDS2GN(Network):
    """2D U-Net with group normalization."""

    """2D U-Net with group normalization."""

    def setup(self):
        print ('2D UNet with 32 channel output')
        base_filter = 32
        (self.feed('data')
        .conv_gn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv1_0')
        .conv_gn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv2_0')
        .conv_gn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv3_0')
        .conv_gn(3, base_filter * 16, 2, center=True, scale=True, name='2dconv4_0'))

        (self.feed('data')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv0_1')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv0_2'))

        (self.feed('2dconv1_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_1')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_2'))

        (self.feed('2dconv2_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_1')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_2'))

        (self.feed('2dconv3_0')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_1')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_2'))

        (self.feed('2dconv4_0')
        .conv_gn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_1')
        .conv_gn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_2')
        .deconv_gn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv5_0'))

        (self.feed('2dconv5_0', '2dconv3_2')
        .concat(axis=-1, name='2dconcat5_0')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_1')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_2')
        .deconv_gn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv6_0'))

        (self.feed('2dconv6_0', '2dconv2_2')
        .concat(axis=-1, name='2dconcat6_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_1')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_2')
        .deconv_gn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv7_0'))

        (self.feed('2dconv7_0', '2dconv1_2')
        .concat(axis=-1, name='2dconcat7_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_1')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_2')
        .deconv_gn(3, base_filter, 2, center=True, scale=True, name='2dconv8_0'))

        (self.feed('2dconv8_0', '2dconv0_2')
        .concat(axis=-1, name='2dconcat8_0')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv8_1')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv8_2')   # end of UNet
        )

class UNetDS2BN(Network):
    """2D U-Net with batch normalization."""

    """2D U-Net with batch normalization."""

    def setup(self):
        print ('2D UNet with 32 channel output')
        base_filter = 16
        (self.feed('data')
        .conv_bn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv1_0')
        .conv_bn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv2_0')
        .conv_bn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv3_0')
        .conv_bn(3, base_filter * 16, 2, center=True, scale=True, name='2dconv4_0'))

        (self.feed('data')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='2dconv0_1')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='2dconv0_2'))

        (self.feed('2dconv1_0')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_1')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_2'))

        (self.feed('2dconv2_0')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_1')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_2'))

        (self.feed('2dconv3_0')
        .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_1')
        .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_2'))

        (self.feed('2dconv4_0')
        .conv_bn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_1')
        .conv_bn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_2')
        .deconv_bn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv5_0'))

        (self.feed('2dconv5_0', '2dconv3_2')
        .concat(axis=-1, name='2dconcat5_0')
        .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_1')
        .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_2')
        .deconv_bn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv6_0'))

        (self.feed('2dconv6_0', '2dconv2_2')
        .concat(axis=-1, name='2dconcat6_0')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_1')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_2')
        .deconv_bn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv7_0'))

        (self.feed('2dconv7_0', '2dconv1_2')
        .concat(axis=-1, name='2dconcat7_0')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_1')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_2')
        .deconv_bn(3, base_filter, 2, center=True, scale=True, name='2dconv8_0'))

        (self.feed('2dconv8_0', '2dconv0_2')
        .concat(axis=-1, name='2dconcat8_0')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='2dconv8_1')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='2dconv8_2')   # end of UNet
        )



class UNetDS2BN_1(Network):
    """2D U-Net with batch normalization."""

    """2D U-Net with batch normalization."""

    def setup(self):
        print ('2D UNet with 32 channel output')
        base_filter = 32
        (self.feed('data')
        .conv_bn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv1_0')
        .conv_bn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv2_0')
        .conv_bn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv3_0')
        .conv_bn(3, base_filter * 16, 2, center=True, scale=True, name='2dconv4_0'))

        (self.feed('data')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='2dconv0_1')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='2dconv0_2'))

        (self.feed('2dconv1_0')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_1')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_2'))

        (self.feed('2dconv2_0')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_1')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_2'))

        (self.feed('2dconv3_0')
        .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_1')
        .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_2'))

        (self.feed('2dconv4_0')
        .reduce_mean(axis=[1,2],keepdims=True)
        .tile([1,FLAGs.max_h//16,FLAGs.max_w//16,1],name='global_feature')
        )

        (self.feed('2dconv4_0','global_feature')
        .concat(axis=-1,name='2dconcat4_0')
        .conv_bn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_1')
        .conv_bn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_2')
        .deconv_bn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv5_0'))

        (self.feed('2dconv5_0', '2dconv3_2')
        .concat(axis=-1, name='2dconcat5_0')
        .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_1')
        .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_2')
        .deconv_bn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv6_0'))

        (self.feed('2dconv6_0', '2dconv2_2')
        .concat(axis=-1, name='2dconcat6_0')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_1')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_2')
        .deconv_bn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv7_0'))

        (self.feed('2dconv7_0', '2dconv1_2')
        .concat(axis=-1, name='2dconcat7_0')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_1')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_2')
        .deconv_bn(3, base_filter, 2, center=True, scale=True, name='2dconv8_0'))

        (self.feed('2dconv8_0', '2dconv0_2')
        .concat(axis=-1, name='2dconcat8_0')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='2dconv8_1')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='2dconv8_2')   # end of UNet
        )


class IterNet(Network):
    def setup(self):
        base_filter = 8
        (self.feed('data')
        .conv_gn(3,base_filter*4,1, center=True, scale=True, name='iter_conv_0')
        .conv_gn(3,base_filter*4,1, center=True, scale=True, name='iter_conv_1')
        .conv_gn(3,base_filter*4,1, center=True, scale=True,relu=False, name='iter_conv_2')
        )
class SNetDS2GN(Network):
    """2D U-Net with group normalization."""

    def setup(self):
        print ('2D SNet with 32 channel output')
        base_filter = 16
        (self.feed('data')
         .conv_bn(3,base_filter,1,dilation_rate=1,center=True,scale=True,name="sconv0_0")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True, name="sconv0_1")
         .conv_bn(3, base_filter*2, 1,dilation_rate=2, center=True, scale=True, name="sconv0_2")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv0_3")
         )
        (self.feed('sconv0_2')
         .conv_bn(3, base_filter*2, 1,dilation_rate=3, center=True, scale=True, name="sconv1_2")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv1_3")
         )
        (self.feed('sconv0_2')
        .conv_bn(3, base_filter*2, 1,dilation_rate=4, center=True, scale=True, name="sconv2_2")
        .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv2_3")
        )
        (self.feed('sconv0_3','sconv1_3','sconv2_3')
        .concat(axis=-1,name='sconcat')
        # .convs_bn(3,base_filter*2,1,dilation_rate=1, center=True, scale=True,relu=True, name='sconv3_0')
        .conv(3,base_filter*2,1,relu=False,name='sconv3_0')
        )
class SNetDS2GN_1(Network):
    """2D U-Net with group normalization."""

    def setup(self):
        print ('2D SNet with 32 channel output')
        base_filter = 16
        (self.feed('data')
         .conv_gn(3,base_filter,1,dilation_rate=1,center=True,scale=True,name="sconv0_0")
         .conv_gn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True, name="sconv0_1")
         .conv_gn(3, base_filter*2, 1,dilation_rate=2, center=True, scale=True, name="sconv0_2")
         .conv_gn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv0_3")
         )
        (self.feed('sconv0_2')
         .conv_gn(3, base_filter*2, 1,dilation_rate=3, center=True, scale=True, name="sconv1_2")
         .conv_gn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv1_3")
         )
        (self.feed('sconv0_2')
        .conv_gn(3, base_filter*2, 1,dilation_rate=4, center=True, scale=True, name="sconv2_2")
        .conv_gn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv2_3")
        )
        (self.feed('sconv0_3','sconv1_3','sconv2_3')
        .concat(axis=-1,name='sconcat')
        # .convs_bn(3,base_filter*2,1,dilation_rate=1, center=True, scale=True,relu=True, name='sconv3_0')
        .conv(3,base_filter*2,1,relu=False,name='sconv3_0')
        )

class SNetDS2BN_base_8(Network):
    """2D U-Net with group normalization."""

    def setup(self):
        print ('2D SNet with 32 channel output')
        base_filter = 8
        (self.feed('data')
         .conv_bn(3,base_filter,1,dilation_rate=1,center=True,scale=True,name="sconv0_0")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True, name="sconv0_1")
         .conv_bn(3, base_filter*2, 1,dilation_rate=2, center=True, scale=True, name="sconv0_2")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv0_3")
         )
        (self.feed('sconv0_2')
         .conv_bn(3, base_filter*2, 1,dilation_rate=3, center=True, scale=True, name="sconv1_2")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv1_3")
         )
        (self.feed('sconv0_2')
        .conv_bn(3, base_filter*2, 1,dilation_rate=4, center=True, scale=True, name="sconv2_2")
        .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv2_3")
        )
        (self.feed('sconv0_3','sconv1_3','sconv2_3')
        .concat(axis=-1,name='sconcat')
        # .convs_bn(3,base_filter*2,1,dilation_rate=1, center=True, scale=True,relu=True, name='sconv3_0')
        .conv(3,base_filter*2,1,relu=False,name='sconv3_0')
        )


class GateNet(Network):
    """network for regularizing 3D cost volume in a encoder-decoder style. Keeping original size."""

    def setup(self):
        # print ('2D with 8 filters')
        base_filter = 8
        (self.feed('data')
        .conv_gn(3, base_filter * 2, 2, center=True, scale=True, name='gateconv1_0')#/2
        .conv_gn(3, base_filter * 4, 2, center=True, scale=True, name='gateconv2_0')#/4
        .conv_gn(3, base_filter * 8, 2, center=True, scale=True, name='gateconv3_0'))#/8

        (self.feed('data')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='gateconv0_1'))

        (self.feed('gateconv1_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='gateconv1_1'))#/2

        (self.feed('gateconv2_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='gateconv2_1'))#/4

        (self.feed('gateconv3_0')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='gateconv3_1')
        .deconv_gn(3, base_filter * 4, 2, center=True, scale=True, name='gateconv4_0'))#/4

        (self.feed('gateconv4_0', 'gateconv2_1')
        .add(name='gateconv4_1')
        .deconv_gn(3, base_filter * 2, 2, center=True, scale=True, name='gateconv5_0'))

        (self.feed('gateconv5_0', 'gateconv1_1')
        .add(name='gateconv5_1')
        .deconv_gn(3, base_filter*1, 2, center=True, scale=True, name='gateconv6_0'))

        (self.feed('gateconv6_0', 'gateconv0_1')
        .add(name='gateconv6_1')
        .conv(3, 32, 1, biased=False, relu=False, name='gateconv6_2'))

class OutputNet(Network):
    """network for regularizing 3D cost volume in a encoder-decoder style. Keeping original size."""

    def setup(self):
        # print ('2D with 8 filters')
        base_filter = 8
        (self.feed('data')
        .conv_gn(3, base_filter * 2, 2, center=True, scale=True, name='outputconv1_0')#/2
        .conv_gn(3, base_filter * 4, 2, center=True, scale=True, name='outputconv2_0')#/4
        .conv_gn(3, base_filter * 8, 2, center=True, scale=True, name='outputconv3_0'))#/8

        (self.feed('data')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='outputconv0_1'))

        (self.feed('outputconv1_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='outputconv1_1'))#/2

        (self.feed('outputconv2_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='outputconv2_1'))#/4

        (self.feed('outputconv3_0')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='outputconv3_1')
        .deconv_gn(3, base_filter * 4, 2, center=True, scale=True, name='outputconv4_0'))#/4

        (self.feed('outputconv4_0', 'outputconv2_1')
        .add(name='outputconv4_1')
        .deconv_gn(3, base_filter *2, 2, center=True, scale=True, name='outputconv5_0'))

        (self.feed('outputconv5_0', 'outputconv1_1')
        .add(name='outputconv5_1')
        .deconv_gn(3, base_filter*1, 2, center=True, scale=True, name='outputconv6_0'))

        (self.feed('outputconv6_0', 'outputconv0_1')
        .add(name='outputconv6_1')
        .conv(3,16, 1, biased=False, relu=False, name='outputconv6_2'))

class MaskNet(Network):
    """network for regularizing mask cost volume in a encoder-decoder style. Keeping original size."""

    def setup(self):
        print ('mask with 8 filters')
        base_filter = 8
        (self.feed('data')
        .conv_gn(3, base_filter * 2, 2, center=True, scale=True, name='maskconv1_0')#/2
        .conv_gn(3, base_filter * 4, 2, center=True, scale=True, name='maskconv2_0')#/4
        .conv_gn(3, base_filter * 8, 2, center=True, scale=True, name='maskconv3_0'))#/8

        (self.feed('data')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='maskconv0_1'))

        (self.feed('maskconv1_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='maskconv1_1'))#/2

        (self.feed('maskconv2_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='maskconv2_1'))#/4

        (self.feed('maskconv3_0')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='maskconv3_1')
        .deconv_gn(3, base_filter * 4, 2, center=True, scale=True, name='maskconv4_0'))#/4

        (self.feed('maskconv4_0', 'maskconv2_1')
        .add(name='maskconv4_1')
        .deconv_gn(3, base_filter * 2, 2, center=True, scale=True, name='maskconv5_0'))

        (self.feed('maskconv5_0', 'maskconv1_1')
        .add(name='maskconv5_1')
        .deconv_gn(3, base_filter, 2, center=True, scale=True, name='maskconv6_0'))

        (self.feed('maskconv6_0', 'maskconv0_1')
        .add(name='maskconv6_1')
        .conv(3, 2, 1, biased=False, relu=False, name='maskconv6_2'))


########################################################################################
###################### 3D CNNs cost volume regularization network ######################
########################################################################################

class RegNetUS0(Network):
    """network for regularizing 3D cost volume in a encoder-decoder style. Keeping original size."""

    def setup(self):
        print ('3D with 8 filters')
        base_filter = 8
        (self.feed('data')
        .conv_bn(3, base_filter * 2, 2, center=True, scale=True, name='3dconv1_0')#/2
        .conv_bn(3, base_filter * 4, 2, center=True, scale=True, name='3dconv2_0')#/4
        .conv_bn(3, base_filter * 8, 2, center=True, scale=True, name='3dconv3_0'))#/8

        (self.feed('data')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='3dconv0_1'))

        (self.feed('3dconv1_0')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='3dconv1_1'))#/2

        (self.feed('3dconv2_0')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='3dconv2_1'))#/4

        (self.feed('3dconv3_0')
        .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='3dconv3_1')
        .deconv_bn(3, base_filter * 4, 2, center=True, scale=True, name='3dconv4_0'))#/4

        (self.feed('3dconv4_0', '3dconv2_1')
        .add(name='3dconv4_1')
        .deconv_bn(3, base_filter * 2, 2, center=True, scale=True, name='3dconv5_0'))

        (self.feed('3dconv5_0', '3dconv1_1')
        .add(name='3dconv5_1')
        .deconv_bn(3, base_filter, 2, center=True, scale=True, name='3dconv6_0'))

        (self.feed('3dconv6_0', '3dconv0_1')
        .add(name='3dconv6_1')
        .conv(3, 1, 1, biased=False, relu=False, name='3dconv6_2'))

class RefineNet(Network):
    """network for depth map refinement using original image."""

    def setup(self):

        (self.feed('color_image', 'depth_image')
        .concat(axis=3, name='concat_image'))

        (self.feed('concat_image')
        .conv_bn(3, 32, 1, name='refine_conv0')
        .conv_bn(3, 32, 1, name='refine_conv1')
        .conv_bn(3, 32, 1, name='refine_conv2')
        .conv(3, 1, 1, name='refine_conv3'))

        (self.feed('refine_conv3', 'depth_image')
        .add(name='refined_depth_image')
         .conv(3,1,1,relu=False,name='refine_conv4')
         )

class RefineNet_1(Network):
    """network for depth map refinement using original image."""

    def setup(self):


        (self.feed('data')
        .conv_bn(3, 32, 1, name='refine_conv0')
        .conv_bn(3, 32, 1, name='refine_conv1')
        .conv_bn(3, 32, 1, name='refine_conv2')
        .conv(3,2, 1, name='refine_conv3')
        .softmax(name="softmax")
        )
        # (self.feed('softmax', 'depth_images')
        #  .reduce_mul(name='refined_depth_image'))


    # """network for depth map refinement using original image"""

    # def setup(self):
    #     # (self.feed('depth_image')
    #     #  .avg_pool(2,1,name='merge_image')
    #     #  )
    #     (self.feed('color_image', 'depth_images')
    #      .concat(axis=3, name='concat_image'))

    #     (self.feed('concat_image')
    #      .conv_gn(3, 32, 1, name='refine_conv0')
    #      .conv_gn(3, 32, 1, name='refine_conv1')
    #      .conv_gn(3, 32, 1, name='refine_conv2')
    #      .conv(3, 4, 1, relu=False, name='refine_conv3')
    #      .softmax(name="softmax")
    #      )

    #     (self.feed('softmax', 'depth_images')
    #      .reduce_mul(name='refined_depth_image'))
