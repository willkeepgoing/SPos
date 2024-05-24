from PIL import Image
from matplotlib import pyplot as plt
import cv2
import wx  # 图形化界面　conda install wxpython
import test_one_photo
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import Model
from test import *
from pathlib import Path


class ClassificationFrame(wx.Frame):

    def __init__(self, *args, **kw):
        super(ClassificationFrame, self).__init__(*args, **kw)
        pnl = wx.Panel(self)
        self.pnl = pnl
        st = wx.StaticText(pnl, label="朝向识别", pos=(200, 0))
        font = st.GetFont()
        font.PointSize += 10
        font = font.Bold()
        st.SetFont(font)

        btn = wx.Button(pnl, -1, 'select')
        btn.Bind(wx.EVT_BUTTON, self.OnSelect)

        self.makeMenuBar()

        self.CreateStatusBar()
        self.SetStatusText("Welcome to flowers world")

    def makeMenuBar(self):
        fileMenu = wx.Menu()
        helloItem = fileMenu.Append(-1, "&Hello...\tCtrl-H", "Help string shown in status bar for this menu item")
        fileMenu.AppendSeparator()
        exitItem = fileMenu.Append(wx.ID_EXIT)
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "Help")

        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)

    def OnExit(self, event):
        self.Close(True)

    def OnHello(self, event):
        wx.MessageBox("Hello again from wxPython")

    def OnAbout(self, event):
        """Display an About Dialog"""
        wx.MessageBox("This is a wxPython Hello World sample",
                      "About Hello World 2",
                      wx.OK | wx.ICON_INFORMATION)

    def OnSelect(self, event):
        wildcard = "image source(*.jpg)|*.*|" \
                   "Compile Python(*.pyc)|*.pyc|" \
                   "All file(*.*)|*.*"
        dialog = wx.FileDialog(None, "Choose a file", os.getcwd(),
                               "", wildcard, wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            model = Model.get_net()
            checkpoint = torch.load('E:\desktop\classify_sitting_three\checkpoints//resnet101(2022.10.14).pth',
                                    map_location='cpu')
            model.load_state_dict(checkpoint["state_dict"])
            img = cv2.imread(dialog.GetPath())

            result = test_one_photo.test_one_image(img, model)
            result_text = wx.StaticText(self.pnl, label=result, pos=(320, 0))
            font = result_text.GetFont()
            font.PointSize += 8
            result_text.SetFont(font)
            self.initimage(name=dialog.GetPath())

    # 生成图片控件
    def initimage(self, name):
        imageShow = wx.Image(name, wx.BITMAP_TYPE_ANY)
        sb = wx.StaticBitmap(self.pnl, -1, imageShow.ConvertToBitmap(), pos=(200, 100), size=(400, 400))
        return sb


if __name__ == '__main__':
    app = wx.App()
    frm = ClassificationFrame(None, title='flower wolrd', size=(1000, 600))
    frm.Show()
    app.MainLoop()