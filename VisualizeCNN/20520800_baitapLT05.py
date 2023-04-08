import os
import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.models import vgg16, VGG16_Weights

@st.cache(allow_output_mutation=True)
def load_features_and_transform(device):
    trans = VGG16_Weights.IMAGENET1K_FEATURES.transforms()
    model = vgg16(weights="IMAGENET1K_FEATURES").to(device)
    model.eval()
    features = model.features

    model_weights = []
    conv_layers = []
    conv_count = 0

    for feature in features:
        if isinstance(feature, nn.Conv2d):
            conv_count += 1
            conv_layers.append(feature)
            model_weights.append(feature.weight.detach().cpu())

    return conv_layers, model_weights, trans, conv_count

def visTensor(tensor, ch=0, allkernels=False, nrow=10, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    return grid.numpy().transpose((1, 2, 0))

if __name__ == "__main__":
    st.title("Visualize VGG16 Convolutional Neural Network")

    # Get Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    conv_layers, model_weights, trans, conv_count = load_features_and_transform(device)

    image_path = st.file_uploader("Upload Image here")
    if image_path:
        # Get Image
        img = Image.open(image_path)
        img_arr = np.array(img)
        st.image(img_arr, "Image Uploaded", width=224)
        img_trans = trans(img).to(device)


        select_conv = st.sidebar.selectbox(
            "Which Conv2D layer you want to view?",
            [*range(1, conv_count + 1)])

        add_image_size = st.sidebar.radio("Choose an feature maps size to view", 
        ("224 x 224 (default)", "128 x 128", "64 x 64", "32 x 32"), index=2)

        output_feature = []
        for layer in conv_layers:
            img_trans = layer(img_trans)
            output_feature.append(img_trans)

        resize_size = 0
        if add_image_size.startswith("224"):
            resize_size = None
        elif add_image_size.startswith("128"):
            resize_size = 128
        elif add_image_size.startswith("64"):
            resize_size = 64
        else:
            resize_size = 32

        tab1, tab2 = st.tabs(["Feature maps", "Kernels"])
        with tab1:
            # Show grid image
            conv = output_feature[int(select_conv) - 1]
            conv = conv.detach().cpu().numpy()
            conv = np.expand_dims(conv, axis=1)
            conv = conv.reshape(conv.shape[0], conv.shape[2], conv.shape[3], conv.shape[1])
            st.image(conv, clamp=True, width=resize_size)
        with tab2:
            kernel_rows = st.sidebar.radio("Choose number of rows to view", [8, 16, 32], index=0)
            kernel = model_weights[int(select_conv) - 1]
            grid = visTensor(kernel, ch=0, allkernels=False, nrow=kernel_rows)
            st.image(grid, width=750)
    else:
        st.info("You should choose an image")