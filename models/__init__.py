from models.quantum_models import Q1_QNN4EO, Q2_AngleSEL, Q3_DataReupload
from models.fusion_models  import (
    E11_MobileViT_ResNet_Concat,
    E12_MobileViT_ResNet_CrossAttn,
    E13_DenseNet_MobileViT_CrossAttn,
    E14_DenseNet_MobileViT_Concat,
)

QUANTUM_MODELS = {
    'Q1_QNN4EO':       Q1_QNN4EO,
    'Q2_AngleSEL':     Q2_AngleSEL,
    'Q3_DataReupload': Q3_DataReupload,
}

FUSION_MODELS = {
    'E11': E11_MobileViT_ResNet_Concat,
    'E12': E12_MobileViT_ResNet_CrossAttn,
    'E13': E13_DenseNet_MobileViT_CrossAttn,
    'E14': E14_DenseNet_MobileViT_Concat,
}
