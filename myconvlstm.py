import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dropout = 0.0):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.dropout = torch.nn.Dropout3d(dropout)
        self.prev_state = None

    def forward(self, input):
        h_prev, c_prev = self.prev_state
        combined = torch.cat((input, h_prev), dim=1)  # concatenate along channel axis

        
        combined_conv = self.dropout(combined)
        combined_conv = self.conv(combined_conv)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = F.sigmoid(cc_i)
        f = F.sigmoid(cc_f)
        o = F.sigmoid(cc_o)
        g = F.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * F.tanh(c_cur)
        
        self.prev_state = (h_cur, c_cur)
        return h_cur, c_cur

    def init_hidden(self, batch_size, cuda=True):
        state = (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                 Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))
        if cuda:
            state = (state[0].cuda(), state[1].cuda())
        self.prev_state = state
        return state

class basic(nn.Module):
    def __init__(self, in_ch, out_ch, input_size, kernel_size = (3, 3), bias = True, dropout = 0.5):
        super(basic, self).__init__()
        self.bias = bias
        self.conv = ConvLSTMCell(input_size=input_size,
                         input_dim=in_ch,
                         hidden_dim=out_ch,
                         kernel_size=kernel_size,
                         bias=self.bias,
                         dropout = dropout)
        self.hidden_state = None

    def forward(self, x):
        h, c = self.hidden_state
        seq_len = x.size(1)
        output_inner=[]
        for t in range(seq_len):
            h, c = self.conv(input=x[:, t, :, :, :])
            output_inner.append(h)
        layer_output = torch.stack(output_inner, dim=1)

        self.hidden_state = (h, c)
        return layer_output
 
    def init_hidden(self, batch_size, cuda=True):
        self.hidden_state = self.conv.init_hidden(batch_size, cuda = cuda)
        return self.hidden_state

class down(nn.Module):
    def __init__(self, in_ch, out_ch, input_size, kernel_size = (3, 3), bias = True, dropout = 0.5):
        super(down, self).__init__()
        self.bias = bias
        self.maxpool = nn.MaxPool3d(2)
        self.conv = ConvLSTMCell(input_size=input_size,
                         input_dim=in_ch,
                         hidden_dim=out_ch,
                         kernel_size=kernel_size,
                         bias=self.bias,
                         dropout = dropout)
        self.hidden_state = None

    def forward(self, x):
        h, c = self.hidden_state
        seq_len = x.size(1)
        output_inner=[]
        for t in range(seq_len):
            h, c = self.conv(input=x[:, t, :, :, :])
            output_inner.append(h)
        layer_output = torch.stack(output_inner, dim=1)

        self.hidden_state = (h, c)
        layer_output = self.maxpool(layer_output)
        return layer_output
 
    def init_hidden(self, batch_size, cuda=True):
        self.hidden_state = self.conv.init_hidden(batch_size, cuda = cuda)
        return self.hidden_state

class up(nn.Module):
    def __init__(self, in_ch, out_ch, input_size, kernel_size = (3, 3), bilinear=True, bias = True, dropout = 0.5):
        super(up, self).__init__()

        self.bias = bias

        self.conv = ConvLSTMCell(input_size=input_size,
                                 input_dim=in_ch,
                                 hidden_dim=out_ch,
                                 kernel_size=kernel_size,
                                 bias=self.bias,
                                 dropout = dropout)

    def forward(self, x, x2):
        # upsample be4 convlstm
        x = F.upsample(x2, (x2.size(3), x2.size(4)), mode='bilinear')
        h, c = self.hidden_state
        seq_len = x.size(1)
        output_inner=[]
        for t in range(seq_len):
            h, c = self.conv(input=x[:, t, :, :, :])
            output_inner.append(h)
        layer_output = torch.stack(output_inner, dim=1)

        self.hidden_state = (h, c)
        return layer_output
 
    def init_hidden(self, batch_size, cuda=True):
        self.hidden_state = self.conv.init_hidden(batch_size, cuda = cuda)
        return self.hidden_state


class MyConvLSTM(nn.Module):

    def __init__(self, input_size = (440, 500), input_dim = 3, kernel_size = (3 ,3), n_classes = 2,
                 batch_first=True, bias=True, return_all_layers=False, dropout = 0.5):
        super(MyConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.kernel_size = kernel_size
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.has_hidden_state = False

        height, width = input_size

        self.inc   = basic(3, 128, (height, width),
                                kernel_size=kernel_size,
                                bias=bias,
                                dropout = dropout)
        
        self.down1 = down(64, 128, (height, width), kernel_size = kernel_size, bias = bias, dropout = dropout)
        self.down2 = down(128, 256, (height//2, width//2), kernel_size = kernel_size, bias = bias, dropout = dropout)
        self.down3 = down(256, 512, (height//4, width//4), kernel_size = kernel_size, bias = bias, dropout = dropout)

        self.bottleneck = ConvLSTMCell(input_size= (height//8, width//8),
                                input_dim=512,
                                hidden_dim=512,
                                kernel_size=kernel_size,
                                bias=self.bias,
                                dropout = dropout)

        self.up3   = up(512, 256, (height//4, width//4), kernel_size = kernel_size, bias=bias, dropout = dropout)
        self.up2   = up(256, 128, (height//2, width//2), kernel_size = kernel_size, bias=bias, dropout = dropout)
        self.up1   = up(128, 64, (height, width), kernel_size = kernel_size, bias=bias, dropout = dropout)
                
        self.outc  = basic(64, n_classes, (height, width),
                                kernel_size=kernel_size,
                                bias=bias,
                                dropout = dropout)

    def forward(self, input):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input = input.permute(1, 0, 2, 3, 4)


        if self.has_hidden_state is False:
            self.init_state(batch_size=input.size(0))
            self.has_hidden_state = True

        x_in = self.inc(input)
        x_out = self.outc(x_in)

        #x_in = self.inc(input) # 1 channel 64
        #x_down1 = self.down1(x_in) # output 1/2 channel 128
        #x_down2 = self.down1(x_down1) # output 1/4 channel 256
        #x_down3 = self.down3(x_down2) # output 1/8 channel 512
        #x_up3 = self.up3(x_down3, x_down2) + x_down2 # output 1/4 channel 256
        #x_up2 = self.up2(x_up3, x_down1) + x_down1 # output 1/2 channel 128
        #x_up1 = self.up1(x_up2, x_in) + x_in # output 1 channel 64
        #x_out = self.outc(x_up1) # output 1 channel 3
        
        if not self.batch_first:
            x_out = x_out.permute(1, 0, 2, 3, 4)
        return x_out

    def init_state(self, batch_size, cuda = True):
        self.inc.init_hidden(batch_size, cuda)
        self.down1.init_hidden(batch_size, cuda)
        self.down2.init_hidden(batch_size, cuda)
        self.down3.init_hidden(batch_size, cuda)
        self.bottleneck.init_hidden(batch_size, cuda)
        self.up1.init_hidden(batch_size, cuda)
        self.up2.init_hidden(batch_size, cuda)
        self.up3.init_hidden(batch_size, cuda)
        self.outc.init_hidden(batch_size, cuda)

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

