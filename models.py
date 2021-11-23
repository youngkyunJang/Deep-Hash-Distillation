from config import *

class Augmentation(nn.Module):
    def __init__(self, org_size, Aw=1.0):
        super(Augmentation, self).__init__()
        self.gk = int(org_size*0.1)
        if self.gk%2==0:
            self.gk += 1
        self.Aug = nn.Sequential(
        Kg.RandomResizedCrop(size=(org_size, org_size), p=1.0*Aw),
        Kg.RandomHorizontalFlip(p=0.5*Aw),
        Kg.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8, hue=0.2, p=0.8*Aw),
        Kg.RandomGrayscale(p=0.2*Aw),
        Kg.RandomGaussianBlur((self.gk, self.gk), (0.1, 2.0), p=0.5*Aw))

    def forward(self, x):
        return self.Aug(x)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()        
        self.F = nn.Sequential(*list(models.alexnet(pretrained=True).features))
        self.Pool = nn.AdaptiveAvgPool2d((6,6))
        self.C = nn.Sequential(*list(models.alexnet(pretrained=True).classifier[:-1]))
    def forward(self, x):
        x = self.F(x)
        x = self.Pool(x)
        x = T.flatten(x, 1)
        x = self.C(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == 'avgpool':
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        x = T.flatten(x, 1)
        return x

class ViT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.pm = timm.create_model(pretrained_name, pretrained=True)
    def forward(self, x):
        x = self.pm.patch_embed(x)
        cls_token = self.pm.cls_token.expand(x.shape[0], -1, -1)
        x = T.cat((cls_token, x), dim=1)
        x = self.pm.pos_drop(x + self.pm.pos_embed)
        x = self.pm.blocks(x)
        x = self.pm.norm(x)
        return x[:, 0]

class DeiT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.pm = timm.create_model(pretrained_name, pretrained=True)
    def forward(self, x):
        x = self.pm.patch_embed(x)
        cls_token = self.pm.cls_token.expand(x.shape[0], -1, -1)
        x = T.cat((cls_token, self.pm.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pm.pos_drop(x + self.pm.pos_embed)
        x = self.pm.blocks(x)
        x = self.pm.norm(x)
        return x[:, 0]

class SwinT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.pm = timm.create_model(pretrained_name, pretrained=True)
    def forward(self, x):
        x = self.pm.patch_embed(x)
        if self.pm.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pm.pos_drop(x)
        x = self.pm.layers(x)
        x = self.pm.norm(x)  # B L C
        x = self.pm.avgpool(x.transpose(1, 2))  # B C 1
        x = T.flatten(x, 1)
        return x