from collections import OrderedDict

from torchvision import models as pt_models
import timm

from robustbench.model_zoo.enums import ThreatModel
from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from .x_vitb16 import create_model as create_model_vitb16
from .x_vitl16 import create_model as create_model_vitl16

from .x_vitb16_rem import create_model_rem as create_model_vitb16_rem
from .x_vitb16_cmae import create_model_cmae as create_model_vitb16_cmae

from .x_vittiny16 import create_model as create_model_vittiny16
from .x_vittiny16_rem import create_model_rem as create_model_vittiny16_rem

mu = (0.485, 0.456, 0.406)
sigma = (0.229, 0.224, 0.225)


linf = OrderedDict(
    [
        ('Wong2020Fast', {  # requires resolution 288 x 288
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1deM2ZNS5tf3S_-eRURJi-IlvUL8WJQ_w',
            'preprocessing': 'Crop288'
        }),
        ('Engstrom2019Robustness', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1T2Fvi1eCJTeAOEzrH_4TAIwO8HTOYVyn',
            'preprocessing': 'Res256Crop224',
        }),
        ('Salman2020Do_R50', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1TmT5oGa1UvVjM3d-XeSj_XmKqBNRUg8r',
            'preprocessing': 'Res256Crop224'
        }),
        ('Salman2020Do_R18', {
            'model': lambda: normalize_model(pt_models.resnet18(), mu, sigma),
            'gdrive_id': '1OThCOQCOxY6lAgxZxgiK3YuZDD7PPfPx',
            'preprocessing': 'Res256Crop224'
        }),
        ('Salman2020Do_50_2', {
            'model': lambda: normalize_model(pt_models.wide_resnet50_2(), mu, sigma),
            'gdrive_id': '1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_R50', {
            'model': lambda: normalize_model(pt_models.resnet50(pretrained=True), mu, sigma),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
    ])

common_corruptions = OrderedDict(
    [
        ('Geirhos2018_SIN', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1hLgeY_rQIaOT4R-t_KyOqPNkczfaedgs',
            'preprocessing': 'Res256Crop224'
        }),
        ('Geirhos2018_SIN_IN', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '139pWopDnNERObZeLsXUysRcLg6N1iZHK',
            'preprocessing': 'Res256Crop224'
        }),
        ('Geirhos2018_SIN_IN_IN', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1xOvyuxpOZ8I5CZOi0EGYG_R6tu3ZaJdO',
            'preprocessing': 'Res256Crop224'
        }),
        ('Hendrycks2020Many', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1kylueoLtYtxkpVzoOA1B6tqdbRl2xt9X',
            'preprocessing': 'Res256Crop224'
        }),
        ('Hendrycks2020AugMix', {
            'model': lambda: normalize_model(pt_models.resnet50(), mu, sigma),
            'gdrive_id': '1xRMj1GlO93tLoCMm0e5wEvZwqhIjxhoJ',
            'preprocessing': 'Res256Crop224'
        }),
        ('Salman2020Do_50_2_Linf', {
            'model': lambda: normalize_model(pt_models.wide_resnet50_2(), mu, sigma),
            'gdrive_id': '1OT7xaQYljrTr3vGbM37xK9SPoPJvbSKB',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_WRN50_2', {
            'model': lambda: normalize_model(pt_models.wide_resnet50_2(pretrained=True), mu, sigma),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_RX50', {
            'model': lambda: normalize_model(pt_models.resnext50_32x4d(pretrained=True), mu, sigma),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_R50', {
            'model': lambda: normalize_model(pt_models.resnet50(pretrained=True), mu, sigma),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_R18', {
            'model': lambda: normalize_model(pt_models.resnet18(pretrained=True), mu, sigma),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_TNT_B', { # working
            'model': lambda: timm.create_model("tnt_b_patch16_224", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_VisFormer_S', { # working
            'model': lambda: timm.create_model("visformer_small", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_VisFormer_T', { # working
            'model': lambda: timm.create_model("visformer_tiny", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_ConvNeXt_L', {
            'model': lambda: timm.create_model("convnext_large", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_ConvNeXt_B', {
            'model': lambda: timm.create_model("convnext_base", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_ConvNeXt_T', {
            'model': lambda: timm.create_model("convnext_tiny", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_ViT_L', {
            'model': lambda: create_model_vitl16("vit_large_patch16_224", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_ViT_B', {
            'model': lambda: create_model_vitb16("vit_base_patch16_224", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_ViT_T', {
            'model': lambda: create_model_vittiny16("vit_tiny_patch16_224", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_VIT_B_REM', {
            'model': lambda: create_model_vitb16_rem("vit_base_patch16_224", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
        ('Standard_VIT_T_REM', {
            'model': lambda: create_model_vittiny16("vit_tiny_patch16_224", pretrained=True),
            'gdrive_id': '',
            'preprocessing': 'Res256Crop224'
        }),
    ])

imagenet_models = OrderedDict([(ThreatModel.Linf, linf),
                               (ThreatModel.corruptions, common_corruptions)])


