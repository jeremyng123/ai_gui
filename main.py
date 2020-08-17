import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QPushButton, \
    QFileDialog, QErrorMessage, QDialog
from PyQt5.QtGui import QIcon, QPixmap
import PyQt5.QtCore as QtCore
import torch
from torch import nn
from torchvision.transforms import Compose
from torchvision import transforms, utils
from PIL import Image
import os.path as osp
import PIL.Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2
from scipy import stats
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#jinyan's model
from segnet import *

########## Louth's Model ##########
def load_model(file_path):
    class conv_block(nn.Module):
        def __init__(self, in_c, out_c):
            super(conv_block, self).__init__()

            self.in_c = in_c
            self.out_c = out_c

            self.conv1 = nn.Conv2d(self.in_c, self.out_c, (3, 3), padding=1)
            self.conv2 = nn.Conv2d(self.out_c, self.out_c, (3, 3), padding=1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x

    class UNet(nn.Module):
        def __init__(self, num_specific_layer=2, in_c=3, out_c=8, c_values=[16, 32, 64, 128, 256]):
            super(UNet, self).__init__()

            self.encoding_layers = nn.ModuleList([conv_block(in_c, c_values[0])])
            for i in range(len(c_values) - 1):
                self.encoding_layers.append(conv_block(c_values[i], c_values[i + 1]))
            # The last encoding layers does not have a maxpool

            # these are the common decoding layers
            self.decoding_layers = nn.ModuleList()
            for i in range(1, num_specific_layer):
                self.decoding_layers.append(nn.ConvTranspose2d(c_values[-i - 1] * 2, c_values[-i - 1], (2, 2), 2))
                self.decoding_layers.append(conv_block(c_values[-i - 1] * 2, c_values[-i - 1]))

            # These are specific to the segmentation output
            self.seg_layers = nn.ModuleList()
            for i in range(num_specific_layer, len(c_values)):
                self.seg_layers.append(nn.ConvTranspose2d(c_values[-i - 1] * 2, c_values[-i - 1], (2, 2), 2))
                self.seg_layers.append(conv_block(c_values[-i - 1] * 2, c_values[-i - 1]))

            # These are specific to the outline drawing
            self.outline_layers = nn.ModuleList()
            for i in range(num_specific_layer, len(c_values)):
                self.outline_layers.append(nn.ConvTranspose2d(c_values[-i - 1] * 2, c_values[-i - 1], (2, 2), 2))
                self.outline_layers.append(conv_block(c_values[-i - 1] * 2, c_values[-i - 1]))

            self.seg_output = nn.Conv2d(c_values[0], out_c, (1, 1))
            self.outline_output = nn.Conv2d(c_values[0], 1, (1, 1))

        def forward(self, x):
            # Encoding layers
            outputs = []
            for encoding_layer in self.encoding_layers[:-1]:
                x = encoding_layer(x)
                outputs.append(x)
                x = F.max_pool2d(x, (2, 2), 2)
            x = self.encoding_layers[-1](x)

            # Common Decoding Layers
            common = len(self.decoding_layers) // 2
            for i in range(common):
                x = self.decoding_layers[2 * i](x)
                x = torch.cat([x, outputs[-i - 1]], dim=1)
                x = self.decoding_layers[2 * i + 1](x)

            # Branch for segmentation
            y = x
            for i in range(len(self.seg_layers) // 2):
                y = self.seg_layers[2 * i](y)
                y = torch.cat([y, outputs[-i - 1 - common]], dim=1)
                y = self.seg_layers[2 * i + 1](y)
            y = self.seg_output(y)

            # Branch for the outline
            for i in range(len(self.outline_layers) // 2):
                x = self.outline_layers[2 * i](x)
                x = torch.cat([x, outputs[-i - 1 - common]], dim=1)
                x = self.outline_layers[2 * i + 1](x)
            x = self.outline_output(x)
            x = torch.sigmoid(x)
            return y, x

    model = UNet()
    model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))

    return model


device = "cpu"


def get_overlay(torch_image, model):
    # This commented out section can open the image from the filename and then subsquently convert to torch array.
    # pil_image = PIL.Image.open("Train/Images/train_1.png").convert('RGB')
    # pil_image = TF.crop(pil_image, 0, 0, 992, 992)
    # np_image = np.array(pil_image)
    # torch_image = TF.to_tensor(pil_image)
    np_image = (torch_image.numpy().transpose([1, 2, 0]) * 255).astype("uint8")
    torch_image = TF.normalize(torch_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_image = torch_image

    pred_seg, pred_outline = model(torch_image.unsqueeze(0))
    pred_seg_flat = np.argmax(pred_seg[0].cpu().detach(), axis=0).numpy().astype("int16")

    pred_outline[pred_outline < 0.3] = 0
    pred_outline[pred_outline > 0.3] = 1.0
    pred_outline_flat = pred_outline[0].cpu().detach().numpy().squeeze().astype("int16")

    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(pred_outline_flat, cv2.MORPH_OPEN, kernel, iterations=2).astype("uint8")
    pred_outline_flat = cv2.dilate(opening, kernel, iterations=1)

    # watershed
    x = pred_seg_flat - (100 * pred_outline_flat)
    x[x < 0] = 0
    y = x.squeeze().astype("uint8")  #
    y[y > 0] = 1
    opening = cv2.morphologyEx(y, cv2.MORPH_OPEN, kernel, iterations=1).astype("uint8")
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 1, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0
    z = np.array([y, y, y]).transpose([1, 2, 0])
    markers = cv2.watershed(z, markers)

    # cls_map = {1:"other", 2:"inflammatory", 3:"epithelial", 4:"epithelial", 5:"spindle-shaped", 6:"spindle-shaped", 7:"spindle-shaped"}
    # cell_count_dict = {"other":0, "inflammatory":0, "epithelial":0, "spindle-shaped":0}
    # actual_cell_count_dict = {"other":0, "inflammatory":0, "epithelial":0, "spindle-shaped":0}

    colors = {1: (255, 255, 51), 2: (204, 0, 204), 3: (0, 255, 0), 4: (255, 0, 0), 5: (0, 0, 204), 6: (102, 102, 255),
              7: (255, 128, 0)}

    for i in range(2, np.max(markers) + 1):

        cls = stats.mode(pred_seg_flat[(markers == i) & (pred_seg_flat != 0)]).mode
        if cls.size == 0:
            pass
        else:
            inst_mask = (markers == i).astype("int16")

            inst_mask = np.pad(inst_mask, 1)
            x = inst_mask[:-1, :] - inst_mask[1:, :]
            y1 = inst_mask[1:, :] - inst_mask[:-1, :]
            sideways = np.logical_or(x[1:], y1[:-1])[:, 1:-1]
            x = inst_mask[:, :-1] - inst_mask[:, 1:]
            y1 = inst_mask[:, 1:] - inst_mask[:, :-1]
            top_bottom = np.logical_or(x[:, 1:], y1[:, :-1])[1:-1]
            inst_mask = np.logical_or(sideways, top_bottom).astype("int16")

            # plt.imshow(inst_mask)
            # plt.show()
            np_image[inst_mask == 1] = colors[int(cls)]

    image = np_image
    cv2.imwrite("temp.jpg", image)

    return "temp.jpg", np.max(markers) - 1


class App(QWidget):

    def __init__(self):
        super(App, self).__init__()
        self.models = {}
        self.labels = []
        self.originalImage = None
        self.initUI()

    def initButtons(self, text, slot):
        '''
        create buttons
        :param text: text on button
        :param slot: respond function
        :return: QPushButton widget
        '''
        button = QPushButton(text)
        button.clicked.connect(slot)
        return button

    def initVBox(self, *args, **kwargs):
        '''
        For buttons. The buttons will be created vertically,
        on the left side of the main window
        :param args: load the various buttons into the VBox layout
        :return: VBoxLayout
        '''
        v_window = QVBoxLayout()
        for arg in args:
            v_window.addWidget(arg)
        if kwargs is not None:
            for key, arg in kwargs.items():
                if key == 'layout':
                    v_window.addLayout(arg)
                else:
                    v_window.addWidget(arg)
        print('return?')
        return v_window

    def initHBox(self, **kwargs):
        '''
        Buttons on the left in vertical box layout
        followed by 2 image space (1000x1000, resize? each to the right)
            left image = original image
            right image = model result

        :param args: load the various widgets into the HBox layout
        :return: HBoxLayout
        '''
        h_window = QHBoxLayout()
        for key, args in kwargs.items():
            if key == 'layout':
                h_window.addLayout(args)
            elif key == 'widget':
                for arg in args:
                    h_window.addWidget(arg)
            else:
                h_window.addLayout(args['original'])
                try:
                    h_window.addLayout(args['model1'])
                except KeyError:
                    pass  # do nothing
                finally:
                    try:
                        h_window.addLayout(args['model2'])
                    except KeyError:
                        pass  # do nothing
        return h_window

    def initUI(self, **kwargs):
        self.setWindowTitle("CoNSeP model")

        # Create buttons
        chooseModel1 = self.initButtons("Choose Model 1", self.pushed_chooseModel1)
        chooseModel2 = self.initButtons("Choose Model 2", self.pushed_chooseModel2)
        chooseImage = self.initButtons("Choose Image", self.pushed_chooseImage)
        evaluate = self.initButtons("Evaluate", self.pushed_evaluate)

        # Place buttons in vertical Box layout
        self.buttons = self.initVBox(chooseModel1, chooseModel2, chooseImage, evaluate)

        # Create image labels
        originalImageWord = QLabel("Original Image:")
        originalImageWord.setAlignment(QtCore.Qt.AlignLeft)
        if self.originalImage is None:
            self.labels = {}
            self.originalImage = QLabel("Original Image")
            self.originalImage.setFixedSize(256, 256)
            self.originalImage.setAlignment(QtCore.Qt.AlignCenter)
            self.originalImage.setStyleSheet("border: 1px solid black")
            originalImage = self.initVBox(originalImageWord, self.originalImage)
            self.labels['original'] = originalImage
        if kwargs is not None:
            for key, arg in kwargs.items():
                self.labels[key] = arg
        self.model1_name = "Model 1"
        self.model1_label = QLabel("Model 1")
        self.model1_label.setAlignment(QtCore.Qt.AlignLeft)
        self.modelImage1 = QLabel(self.model1_name + " Image")
        self.modelImage1.setFixedSize(256, 256)
        self.modelImage1.setAlignment(QtCore.Qt.AlignCenter)
        self.modelImage1.setStyleSheet("border: 1px solid black")
        additionalOutput = QLabel("Number of cells: ")
        self.additionalOutputValue1 = QLabel("")
        hbox = self.initHBox(widget=[additionalOutput, self.additionalOutputValue1])
        model1Images = self.initVBox(self.model1_label, self.modelImage1, layout=hbox)
        self.labels['model1'] = model1Images

        self.model2_name = "Model 2"
        self.model2_label = QLabel("Model 2")
        self.model2_label.setAlignment(QtCore.Qt.AlignLeft)
        self.modelImage2 = QLabel(self.model2_name + " Image")
        self.modelImage2.setFixedSize(256, 256)
        self.modelImage2.setAlignment(QtCore.Qt.AlignCenter)
        self.modelImage2.setStyleSheet("border: 1px solid black")
        additionalOutput = QLabel("Number of cells: ")
        self.additionalOutputValue2 = QLabel("")
        hbox = self.initHBox(widget=[additionalOutput, self.additionalOutputValue2])
        model2Images = self.initVBox(self.model2_label, self.modelImage2, layout=hbox)
        self.labels['model2'] = model2Images

        # place all labels/layout into a horizontal Box layout
        self.h_window = self.initHBox(layout=self.buttons, labels=self.labels)
        self.setLayout(self.h_window)

        self.show()

    def pushed_chooseModel1(self):
        # TODO: opens a dialog to choose model file
        model_filter = "pytorch Models (*.pth *.pt)"
        fileName, _ = QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath(), model_filter)
        if fileName == "":
            return
        self.model1_name = fileName.split('/')[-1]
        self.model1_label.setText(self.model1_name)
        print(fileName)
        if self.model1_name == "best_weights.pth":
            self.models[self.model1_name] = load_model(fileName)  # torch.load(fileName, map_location=torch.device('cpu'))
        else:
            segnet1 = segnet()
            segnet1 = segnet1.load(fileName)
            # segnet1.eval()
            self.models[self.model1_name] = segnet1

    def pushed_chooseModel2(self):
        # TODO: opens a dialog to choose model file
        model_filter = "pytorch Models (*.pth *.pt)"
        fileName, _ = QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath(), model_filter)
        if fileName == "":
            return
        print(fileName)
        self.model2_name = fileName.split('/')[-1]
        self.model2_label.setText(self.model2_name)
        if self.model2_name == "best_weights.pth":
            self.models[self.model2_name] = load_model(fileName)  # torch.load(fileName, map_location=torch.device('cpu'))
        else:
            segnet2 = segnet()
            segnet2 = segnet2.load(fileName)
            # segnet2.eval()
            self.models[self.model2_name] = segnet2

    def pushed_chooseImage(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath(), '*.png')
        pixmap = QPixmap(self.fileName).scaled(256, 256, QtCore.Qt.KeepAspectRatio)
        self.originalImage.setPixmap(pixmap)

    def pushed_evaluate(self):
        try:
            # TODO: with model and image selected, parse image into
            transform = Compose([transforms.Resize(256),
                                 transforms.ToTensor(),
                                 ])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                          std=[0.229, 0.224, 0.225])
            try:
                image = Image.open(self.fileName).convert('RGB')
                image = transform(image)
            except AttributeError:
                print("Attribute Error: did not assign self.fileName!")
                return

            try:
                model_image1 = None
                model_image2 = None
                if self.model2_name is not None and self.model1_name is not None:  # self.model1_name is not None and
                    if self.model2_name == "best_weights.pth":
                        image1 = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        modelimage1 = self.models[self.model1_name](image1.unsqueeze(0))
                        modelimage1 = modelimage1.to(dtype=float)
                        pred_c = torch.argmax(modelimage1, 1).numpy()
                        # print(pred_c)
                        plt.imsave("image.png", pred_c.squeeze())
                        model_image1 = "image.png"
                        ans2 = get_overlay(image, self.models[self.model2_name])
                        # if len(ans1) > 1:
                        #     model_image1 = ans1[0]
                        #     self.additionalOutputValue = ans1[1]
                        #     model_image2 = ans2
                        # else:
                        model_image2 = ans2[0]
                        self.additionalOutputValue2.setText(str(ans2[1]))
                    else:
                        image2 = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        modelimage2 = self.models[self.model2_name](image2.unsqueeze(0))
                        modelimage2 = modelimage2.to(dtype=float)
                        pred_c = torch.argmax(modelimage2, 1).numpy()
                        # print(pred_c)
                        plt.imsave("image.png", pred_c.squeeze())
                        model_image2 = "image.png"
                        ans1 = get_overlay(image, self.models[self.model1_name])
                        # if len(ans1) > 1:
                        #     model_image1 = ans1[0]
                        #     self.additionalOutputValue = ans1[1]
                        #     model_image2 = ans2
                        # else:
                        model_image1 = ans1[0]
                        self.additionalOutputValue1.setText(str(ans1[1]))

                pixmap1 = QPixmap(model_image1).scaled(256, 256, QtCore.Qt.KeepAspectRatio)
                pixmap2 = QPixmap(model_image2).scaled(256, 256, QtCore.Qt.KeepAspectRatio)
                self.modelImage1.setPixmap(pixmap1)
                self.modelImage2.setPixmap(pixmap2)

            except AttributeError as e:
                error_dialog = QErrorMessage()
                error_dialog.showMessage(e)
        except:
            e = sys.exc_info()[0]
            error_dialog = QErrorMessage()
            error_dialog.showMessage(e)


if __name__ == '__main__':
    # model_state_dict = torch.load("C:/Users/ngyzj/Documents/sutd/50.021 - AI/groupProject/gui/weights_e30_.pt", map_location=torch.device('cpu'))
    # image = Image.open("C:/Users/ngyzj/Documents/sutd/50.021 - AI/groupProject/Train/Images/train_1.png").convert('RGB')
    # transform = Compose([transforms.Resize(256),
    #                      transforms.ToTensor(),
    #                      ])
    # image = transform(image)
    # ans1_state_dict = model_state_dict
    # ans1 = segnet()
    # ans1.load_state_dict(ans1_state_dict)
    # ans1.eval()
    # image1 = image.unsqueeze(0)
    # model_image1 = ans1(image1)
    # model_image1 = torch.argmax(model_image1, 1).cpu()[0]
    # plt.imsave("image.png",model_image1)

    app = QApplication(sys.argv)
    mainWindow = App()
    mainWindow.show()

    sys.exit(app.exec_())