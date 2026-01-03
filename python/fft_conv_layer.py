import torch
import fft_conv_cuda

class FFTConv2D(torch.nn.Module):
    def __init__(self, weight, bias=None, padding_mode=0):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.padding_mode = padding_mode  # SAME = 0

    def forward(self, x):
        batch, channels, h, w = x.shape
        out_channels = self.weight.shape[0]

        outputs = []

        for b in range(batch):
            out_maps = []
            for oc in range(out_channels):
                acc = None
                for ic in range(channels):
                    result = fft_conv_cuda.forward(
                        x[b, ic].cpu().numpy(),
                        self.weight[oc, ic].cpu().numpy(),
                        self.padding_mode
                    )
                    acc = result if acc is None else acc + result
                if self.bias is not None:
                    acc += self.bias[oc].item()
                out_maps.append(torch.from_numpy(acc))
            outputs.append(torch.stack(out_maps))

        return torch.stack(outputs).to(x.device)
