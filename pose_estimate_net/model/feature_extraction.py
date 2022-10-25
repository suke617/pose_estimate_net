import torch
import torchvision.models as models


class Model_Select():
    model_resnet50 = models.resnet50(pretrained=True)
    #最終層の付け替え
    model_resnet50.fc = torch.nn.Linear(model_resnet50.fc.in_features, 10)

    #AlexNet
    alexnet = models.alexnet()

    #GoogLeNet
    googlenet = models.googlenet()

    #MobileNetV3
    mobilenet_v3_large = models.mobilenet_v3_large()

    #EfficientNet
    efficientnet_b7 = models.efficientnet_b7()
    efficientnet_b7.classifier = torch.nn.Linear(in_features=2560 , out_features=10)


    efficientnet_v2 = models.efficientnet_v2_s()
    print(efficientnet_v2)

    # model_name = 1
    # if model_name == 1 :
    #     mdoel = model_resnet50

    # elif model_name == 2 :
    #     model = alexnet

    # elif model_name == 3 :
    #     model = googlenet

    # elif model_name == 4 :
    #     model = mobilenet_v3_large

    # elif model_name == 5 :
    #     model = efficientnet_b7

    # else :
    #     model = efficientnet_v2
