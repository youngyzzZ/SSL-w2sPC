from networks.unet import UNet, UNet_DS, UNet_URPC, UNet_CCT, UNet_W2S
# from unet import UNet, UNet_DS, UNet_URPC, UNet_CCT, UNet_W2S

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == 'unet_w2s':
        net = UNet_W2S(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net


if __name__ == '__main__':
    model = net_factory('unet_w2s')
    print(model)
