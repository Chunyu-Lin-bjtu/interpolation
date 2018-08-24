### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import matplotlib.pyplot as plt
import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
import re
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.lossG_LAP = []
        self.lossG_COLOR = []
        self.lossG_VGG = []
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.log_lap_png_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log_lap.png')
        self.log_color_png_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log_color.png')
        self.log_vgg_png_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log_vgg.png')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=5)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: read pretrained errors
    def read_pretrained_errors(self):
        with open(self.log_name, "r") as log_file:
            #print log_file.read()
            for line in log_file.readlines():
                lossG_LAP=re.findall(r"G_LAP: (.+?) G_COLOR",line)
                lossG_VGG=re.findall(r"G_VGG: (.+?) ",line)
                lossG_COLOR=re.findall(r"G_COLOR: (.+?) G_VGG ",line)
                if(lossG_LAP and lossG_VGG and lossG_COLOR):
                    self.lossG_COLOR.append(float(lossG_COLOR[0]))
                    self.lossG_LAP.append(float(lossG_LAP[0]))
                    self.lossG_VGG.append(float(lossG_VGG[0]))
		

    # errors: draw error figure
    def plot_draw_errors(self, errors, step):
        for tag, value in errors.items():
                if(tag == 'G_COLOR'):
                    self.lossG_COLOR.append(value)
                if(tag == 'G_LAP'):
                    self.lossG_LAP.append(value)
                if(tag == 'G_VGG'):
                    self.lossG_VGG.append(value)
        matG_COLOR=np.array(self.lossG_COLOR)
        matG_LAP=np.array(self.lossG_LAP)
        matG_VGG=np.array(self.lossG_VGG)
        _, ax1 = plt.subplots()
        ax1.plot(np.arange(len(matG_COLOR)), matG_COLOR, 'y')
        #ax1.plot(np.arange(len(matG_LAP)), matG_LAP, 'r')
        #ax1.plot(np.arange(len(matG_VGG)), matG_VGG, 'b')
        ax1.set_xlabel('iteration')  
        ax1.set_ylabel('loss')
        plt.savefig(self.log_color_png_name)
        plt.close('all')
        #
        _, ax1 = plt.subplots()
        #ax1.plot(np.arange(len(matG_COLOR)), matG_COLOR, 'y')
        ax1.plot(np.arange(len(matG_LAP)), matG_LAP, 'r')
        #ax1.plot(np.arange(len(matG_VGG)), matG_VGG, 'b')
        ax1.set_xlabel('iteration')  
        ax1.set_ylabel('loss')
        plt.savefig(self.log_lap_png_name)
        plt.close('all')
        #
        _, ax1 = plt.subplots()
        #ax1.plot(np.arange(len(matG_COLOR)), matG_COLOR, 'y')
        #ax1.plot(np.arange(len(matG_LAP)), matG_LAP, 'r')
        ax1.plot(np.arange(len(matG_VGG)), matG_VGG, 'b')
        ax1.set_xlabel('iteration')  
        ax1.set_ylabel('loss')
        plt.savefig(self.log_vgg_png_name)
        plt.close('all')

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    # save image to the disk
    def my_save_images(self, visuals, epoch, step):
        #if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join('/home/george/project/pix2pixHD_3.0/results/label2city_512p/test_latest/40/', 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join('/home/george/project/pix2pixHD_3.0/results/label2city_512p/test_latest/40/', 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML('/home/george/project/pix2pixHD_3.0/results/label2city_512p/test_latest/web/', 'Experiment name = %s' % self.name, refresh=5)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()
