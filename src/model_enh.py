import torch
from collections import OrderedDict

import torch
import torch.nn as nn

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([
                      ('conv1_1', [3, 64, 3, 1, 1]),
                      ('conv1_2', [64, 64, 3, 1, 1]),
                      ('pool1_stage1', [2, 2, 0]),
                      ('conv2_1', [64, 128, 3, 1, 1]),
                      ('conv2_2', [128, 128, 3, 1, 1]),
                      ('pool2_stage1', [2, 2, 0]),
                      ('conv3_1', [128, 256, 3, 1, 1]),
                      ('conv3_2', [256, 256, 3, 1, 1]),
                      ('conv3_3', [256, 256, 3, 1, 1]),
                      ('conv3_4', [256, 256, 3, 1, 1]),
                      ('pool3_stage1', [2, 2, 0]),
                      ('conv4_1', [256, 512, 3, 1, 1]),
                      ('conv4_2', [512, 512, 3, 1, 1]),
                      ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                      ('conv4_4_CPM', [256, 128, 3, 1, 1])
                  ])


        # Stage 1
        block1_1 = OrderedDict([
                        ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
                    ])

        block1_2 = OrderedDict([
                        ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
                    ])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2


        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                    ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
                ])

            blocks['block%d_2' % i] = OrderedDict([
                    ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
                ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']



    def forward(self, x):
        
        out1 = self.model0(x)
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        outs = [out1_1, out1_2]
        outs.extend([out2_1, out2_2])
        outs.extend([out3_1, out3_2])
        outs.extend([out4_1, out4_2])
        outs.extend([out5_1, out5_2])
        outs.extend([out6_1, out6_2])

        return out1, out6_1, out6_2



class add_model(nn.Module):
    def __init__(self):
        super(add_model, self).__init__()

        layers = []
        layer = nn.Upsample(size=(92, 164), scale_factor=None, mode='nearest', align_corners=None,)
        layers.append(("add_upsampling_1", layer))
        layer = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)     
        layers.append(("add_conv2d_1", layer))
        layers.append(('add_relu_1', nn.ReLU(inplace=True)))
        layer = nn.Upsample(size=(135, 240), scale_factor=None, mode='nearest', align_corners=None,)
        layers.append(("add_upsampling_1", layer))
        layer = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)     
        layers.append(("add_conv2d_1", layer))
        layers.append(('add_relu_1', nn.ReLU(inplace=True)))
        self.add_feature_model = nn.Sequential(OrderedDict(layers))

        no_relu_layers = ['add_conv5_5_CPM_L1', 'add_conv5_5_CPM_L2', 'add_Mconv7_stage2_L1',\
                          'add_Mconv7_stage2_L2', 'add_Mconv7_stage3_L1', 'add_Mconv7_stage3_L2',\
                          'add_Mconv7_stage4_L1', 'add_Mconv7_stage4_L2', 'add_Mconv7_stage5_L1',\
                          'add_Mconv7_stage5_L2', 'add_Mconv7_stage6_L1', 'add_Mconv7_stage6_L1']

        add_blocks = {}

        add_block1_1 = OrderedDict([
                        ('add_conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
                        ('add_conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                        ('add_conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                        ('add_conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                        ('add_conv5_5_CPM_L1', [512, 8, 1, 1, 0])
                    ])

        add_block1_2 = OrderedDict([
                        ('add_conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
                        ('add_conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                        ('add_conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
                        ('add_conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                        ('add_conv5_5_CPM_L2', [512, 6, 1, 1, 0])
                    ])
        add_blocks['add_block1_1'] = add_block1_1
        add_blocks['add_block1_2'] = add_block1_2


        for i in range(2, 7):
            add_blocks['add_block%d_1' % i] = OrderedDict([
                    ('add_Mconv1_stage%d_L1' % i, [142, 128, 7, 1, 3]),
                    ('add_Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('add_Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('add_Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('add_Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('add_Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                    ('add_Mconv7_stage%d_L1' % i, [128, 8, 1, 1, 0])
                ])

            add_blocks['add_block%d_2' % i] = OrderedDict([
                    ('add_Mconv1_stage%d_L2' % i, [142, 128, 7, 1, 3]),
                    ('add_Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('add_Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('add_Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('add_Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('add_Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                    ('add_Mconv7_stage%d_L2' % i, [128, 6, 1, 1, 0])
                ])

        for k in add_blocks.keys():
            add_blocks[k] = make_layers(add_blocks[k], no_relu_layers)

        self.add_model1_1 = add_blocks['add_block1_1']
        self.add_model2_1 = add_blocks['add_block2_1']
        self.add_model3_1 = add_blocks['add_block3_1']
        self.add_model4_1 = add_blocks['add_block4_1']
        self.add_model5_1 = add_blocks['add_block5_1']
        self.add_model6_1 = add_blocks['add_block6_1']

        self.add_model1_2 = add_blocks['add_block1_2']
        self.add_model2_2 = add_blocks['add_block2_2']
        self.add_model3_2 = add_blocks['add_block3_2']
        self.add_model4_2 = add_blocks['add_block4_2']
        self.add_model5_2 = add_blocks['add_block5_2']
        self.add_model6_2 = add_blocks['add_block6_2']


    def forward(self, x):

        out1 = self.add_feature_model(x)
        add_out1_1 = self.add_model1_1(out1)
        add_out1_2 = self.add_model1_2(out1)
        add_out2 = torch.cat([add_out1_1, add_out1_2, out1], 1)
        add_out2_1 = self.add_model2_1(add_out2)
        add_out2_2 = self.add_model2_2(add_out2)
        add_out3 = torch.cat([add_out2_1, add_out2_2, out1], 1)
        add_out3_1 = self.add_model3_1(add_out3)
        add_out3_2 = self.add_model3_2(add_out3)
        add_out4 = torch.cat([add_out3_1, add_out3_2, out1], 1)
        add_out4_1 = self.add_model4_1(add_out4)
        add_out4_2 = self.add_model4_2(add_out4)
        add_out5 = torch.cat([add_out4_1, add_out4_2, out1], 1)
        add_out5_1 = self.add_model5_1(add_out5)
        add_out5_2 = self.add_model5_2(add_out5)
        add_out6 = torch.cat([add_out5_1, add_out5_2, out1], 1)
        add_out6_1 = self.add_model6_1(add_out6)
        add_out6_2 = self.add_model6_2(add_out6)

        add_outs = [add_out1_1, add_out1_2]
        add_outs.extend([add_out2_1, add_out2_2])
        add_outs.extend([add_out3_1, add_out3_2])
        add_outs.extend([add_out4_1, add_out4_2])
        add_outs.extend([add_out5_1, add_out5_2])
        add_outs.extend([add_out6_1, add_out6_2])

        return add_outs


