# Copyright 2018 Egor Zindy - University of Manchester
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from PIL import Image
from scipy.misc import imresize
import fnmatch
import os
import tifffile
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.font_manager as fm


import pandas as pd
import libatrous
import read_roi
import itertools

# https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def get_combos(vec,start=1):
    ret = []
    for i in range(len(vec)):
        if i+1<start:
            continue
        for comb in itertools.combinations(vec, i+1):
            ret.append(comb)

    return ret

################################################################################
#
# GUI
#
################################################################################
import ipywidgets
from IPython.display import display, HTML, clear_output
import traitlets
from ipywidgets import widgets
from tkinter import Tk, filedialog

def make_fn_rois(fn):
    fn_rois = fn+".zip"
    if not os.path.exists(fn_rois):
        fn_rois = fn_rois.replace(".tif",".lif")
        if not os.path.exists(fn_rois):
            fn_rois = None

    return fn_rois

class SelectFolderButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self,directory=None):
        super(SelectFolderButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())

        if directory is None or not os.path.exists(directory):
            directory = os.getcwd()
            self.description = "Select a folder"
            self.icon = "square-o"
            self.style.button_color = "orange"
        else:
            self.description = "Folder selected"
            self.tooltip = directory
            self.icon = "check-square-o"
            self.style.button_color = "lightgreen"

        self.directory = directory

        # Create the button.
        # Set on click behavior.
        self.on_click(self.select_folder)

    @staticmethod
    def select_folder(b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button 
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        b.directory = filedialog.askdirectory(initialdir=b.directory)
        
        if b.directory != "":
            b.description = "Folder selected"
            b.tooltip = b.directory
            b.icon = "check-square-o"
            b.style.button_color = "lightgreen"
        
class GUI(object):
    fig_desc = ['< Image', 'Update', 'Image >', 'Save PDF']
    save_desc = ['Overlap data', 'Manders data', 'RGB PDF', 'Overlap PDF']

    channels_rgb3=[[0,1,4],[0,1,0],[1,0,0],[0,0,1]]
    channel_names=['Blue','Green','Red','FarRed']

    def __init__(self,directory=None,threshold=0.5,debug=False):
        self.channels = [0,1,2,3]
        self.image_index = 0
        self.threshold = threshold
        self.n_images = 0
        self.debug=debug

        self.buttons_rgb = None
        self.buttons_overlap = None

        self.channel_names = ["DAPI","Green_Ref","Red","Far Red"]
        self.channel_values = [("Channel %d" % (c+1),c) for c in range(4)]
        self.directory = directory

        if directory is not None:
            print directory
            self.filenames = get_files(self.directory,sanitized=False)
            if self.debug and len(self.filenames) > 5:
                self.filenames = self.filenames[:5]

            self.n_images = len(self.filenames)
            self.print_info()
        else:
            self.filenames = []

    def print_info(self):
        print "Folder: %s" % self.directory
        print "Found %d files" % len(self.filenames)
        n_rois = 0
        for fn in self.filenames:
            fn_rois = make_fn_rois(fn)
            if fn_rois is not None:
                n_rois+=1
        print "Found %d ROIs" % n_rois

    def display_main(self):
        button = SelectFolderButton(self.directory)
        button.on_click(self.on_button_change)
        display(button)

        #The threshold value
        text = widgets.FloatText(value=self.threshold,description="Threshold", width=200)
        text.observe(self.on_text)
        display(text)

        for c in range(4):
            n = self.channel_names[c]
            dropdown = ipywidgets.Dropdown(
                options=self.channel_values,
                value=c,
                description=n,
                disabled=False,
                width=200
            )
            dropdown.observe(self.on_dropdown_change)
            display(dropdown)

        btns = []
        for desc in self.save_desc:
            btn = widgets.Button(description = desc)
            btn.on_click(self.on_button_save)
            btns.append(btn)
        btns.append(widgets.Label(value=""))
        self.buttons_save = widgets.HBox(btns)
        display(self.buttons_save)

    def on_text(self, change):
        if not 'value' in change['new'].keys():
            return

        threshold = change['new']['value']
        if threshold < 0:
            threshold = 0
            change['owner'].value = 0

        self.threshold = threshold


    def on_button_save(self, change):
        if self.directory is None:
            return

        if change.description == self.save_desc[0]:
            self.create_excels(True)
            #Save overlap
        elif change.description == self.save_desc[1]:
            self.create_excels(False)
            #Save Manders
        elif change.description == self.save_desc[2]:
            #Save RGB
            self.create_pdf(overlap=False)
        elif change.description == self.save_desc[3]:
            #Save Overlap
            self.create_pdf(overlap=True)

    def create_excels(self,overlap=True):
        if self.directory is None:
            return

        fn_out = os.path.join(self.directory, "data_overlap.xlsx")
        if overlap:
            df = self.create_overlap_data()
        else:
            fn_out = fn_out.replace("overlap","manders")
            df = self.create_manders_data()

        print "Writing "+fn_out

        with pd.ExcelWriter(fn_out) as writer:  # doctest: +SKIP
            df.to_excel(writer, sheet_name="Values", index=False, engine='openpyxl')

        #Making an HTML
        fn_out = fn_out.replace("xlsx","html")
        print "Writing "+fn_out

        s = df.style.bar(subset=df.columns[3:], color='#d65f5f').render()
        with open(fn_out,"w") as f:
            f.write(s)

        print "All done!"

    def display_rgb(self):
        if self.filenames == []:
            print "No files loaded, please select a folder first!"
            return

        #The threshold value
        text = widgets.FloatText(value=self.threshold,description="Thresh", width=10)
        text.observe(self.on_text)
        btns = [text]
        for desc in self.fig_desc:
            btn = widgets.Button(description = desc)
            btn.on_click(self.on_button_image)
            btns.append(btn)
        btns.append(widgets.Label(value=""))

        self.buttons_rgb = widgets.HBox(btns)
        #display(self.buttons_rgb)

        self.fig_rgb, axes = plt.subplots(2, 2, figsize=(12,12))
        self.fig_rgb.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0, hspace=0)
        self.axes_rgb = axes.flatten()
        self.update_figure(overlap=False)

    def display_overlap(self):
        if self.filenames == []:
            print "No files loaded, please select a folder first!"
            return

        #The threshold value
        text = widgets.FloatText(value=self.threshold,description="Thresh", width=10)
        text.observe(self.on_text)
        btns = [text]
        for desc in self.fig_desc:
            btn = widgets.Button(description = desc)
            btn.on_click(self.on_button_image)
            btns.append(btn)
        btns.append(widgets.Label(value=""))

        self.buttons_overlap = widgets.HBox(btns)
        #display(self.buttons_overlap)

        self.fig_overlap, axes = plt.subplots(2, 2, figsize=(12,12))
        self.fig_overlap.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0, hspace=0)
        self.axes_overlap = axes.flatten()
        self.update_figure(overlap=True)

    def create_figure(self, fn, fig, axes, overlap=False):
        tif = tifffile.TiffFile(fn)
        arr = tif.asarray()
        orig_size = arr.shape[1]

        arr,was_sanitized = sanitize_stack(arr)

        #extent = lc.get_extent(tif)
        tif.close()
        arr_filtered = get_filtered(arr,threshold=self.threshold)

        #Reading the region of interest
        fn_rois = make_fn_rois(fn)
        if fn_rois is not None:
            roi = read_roi.read_roi_zip(fn_rois)
        else:
            roi = None

        #IGNORE DAPI
        channel_names = [self.channel_names[i] for i in self.channels]
        channels_rgb = [self.channels_rgb3[i] for i in self.channels]
        channels = self.channels[1:]

        if overlap:
            composite_channels = None
        else:
            composite_channels = self.channels[1:]

        #Clearing all the graphs
        for ax in axes:
            ax.cla()

        make_composite(arr_filtered,axes,None,
                channel_names=channel_names, channels=channels,
                composite_channels=composite_channels,channels_rgb=channels_rgb,
                overlap=overlap,threshold=self.threshold)

        if overlap:
            prod = product(arr_filtered,channels)
            name = ''
            for c in channel_names[1:]:
                name += '%s x ' % c
            name = name[:-3]

            display_image(axes[-1],prod,None,cmap='gray_r',text_color='black',
                    name=name,color=None,size=False)


            cmap0 = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', [[0,0,0,0],[0.5,0,0.5,0.5]], 1)

            nz = prod > self.threshold
            im_coloc = np.ma.masked_where(nz == False,prod)
            axes[-1].imshow(im_coloc,cmap=cmap0,vmin=0,vmax=1,origin='lower')

        if roi is not None:
            expand_roi(roi,orig_size=orig_size)
            for i in range(4):
                c = 'k'
                if i == 3 and not overlap:
                    c = 'w'
                add_roi(axes[i],roi.values(),color=c)

        return was_sanitized

    def update_figure(self,overlap=False):
        fn = self.filenames[self.image_index]

        if overlap:
            axes = self.axes_overlap
            buttons = self.buttons_overlap
            fig = self.fig_overlap
        else:
            axes = self.axes_rgb
            buttons = self.buttons_rgb 
            fig = self.fig_rgb

        label = buttons.children[-1]
        was_sanitized = self.create_figure(fn,fig,axes,overlap=overlap)
        plt.close()

        clear_output(wait=True)
        _,f = os.path.split(fn)
        desc = f+" (%d / %d)" % (self.image_index+1,self.n_images)
        if was_sanitized:
            desc = desc + " *Sanitized"
        label.value = desc

        display(buttons)
        display(fig)

    def on_dropdown_change(self,change):
        if change['name'] == 'value' and (change['new'] != change['old']):
            pos = self.channel_names.index(change['owner'].description)
            self.channels[pos] = change['new']

    def on_button_change(self,change):
        if change.directory != "":
            self.directory = change.directory

            self.filenames = get_files(self.directory,sanitized=False)
            if self.debug and len(self.filenames) > 5:
                self.filenames = self.filenames[:5]

            self.n_images = len(self.filenames)

            self.print_info()

    def on_button_image(self,change):
        if self.n_images == 0:
            return

        #By default
        do_overlap = False
        if self.buttons_overlap is not None:
            do_overlap = change in self.buttons_overlap.children

        if change.description == self.fig_desc[3]:
            #Save the PDF
            self.create_pdf(overlap=do_overlap)
            return

        if change.description == self.fig_desc[0] and self.image_index > 0:
            self.image_index -= 1
        elif change.description == self.fig_desc[2] and self.image_index <= self.n_images-2:
            self.image_index += 1

        self.update_figure(overlap=do_overlap)

    def create_pdf(self,overlap=False):
        fn_pdf = os.path.join(self.directory,"figure_rgb.pdf")
        if overlap:
            fn_pdf = fn_pdf.replace("rgb.pdf","overlap.pdf")

        pdf_pages = PdfPages(fn_pdf)

        progress = self.display_progress()
        for fn in self.filenames:
            fig, axes = plt.subplots(2, 2, figsize=(12,12))
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0, hspace=0)
            axes = axes.flatten()
            fig.suptitle(fn,y=0.965)
            was_sanitized = self.create_figure(fn,fig,axes,overlap=overlap)
            pdf_pages.savefig(fig)
            plt.close()

            #Update the progress bar
            progress.value += 1

        print "Writing "+fn_pdf
        pdf_pages.close()

        print "All done!"
        progress.value = 0

    def get_channel_name(self,p,short=False):
        s = self.channel_names[p]
        if short:
            s = s.replace("Far Red","FRed").replace("_Ref","")
        return s

    def get_col_name(self, pair, coloc=False):
        s = ""
        for p in pair:
            delim = "-"
            if coloc:
                delim = "On"
            n_delim = len(delim)
            s += self.get_channel_name(p,short=True)+delim

        s = s[:-n_delim]
        if len(pair) == 1:
            s = s+" prop"
        elif coloc:
            s = s+" coloc"
        else:
            s = s+" overlap"
        return s

    def create_manders_data(self):
        directory = self.directory
        filenames = self.filenames
        n_files = len(filenames)

        columns = ['Folder', 'Filename', 'ROI', 'Area',
                'Green Prop', 'Red Prop', 'FRed Prop',

                'Green-Red First Costes Threshold',
                'Green-Red Second Costes Threshold',
                'RedOnGreen Manders',
                'RedOnGreen Manders (Costes)',
                'GreenOnRed Manders',
                'GreenOnRed Manders (Costes)',

                'Green-FRed First Costes Threshold',
                'Green-FRed Second Costes Threshold',
                'FRedOnGreen Manders',
                'FRedOnGreen Manders (Costes)',
                'GreenOnFRed Manders',
                'GreenOnFRed Manders (Costes)',

                'Red-FRed First Costes Threshold',
                'Red-FRed Second Costes Threshold',
                'FRedOnRed Manders',
                'FRedOnRed Manders (Costes)',
                'RedOnFRed Manders',
                'RedOnFRed Manders (Costes)',

                'Green-RedFRed First Costes Threshold',
                'Green-RedFRed Second Costes Threshold',
                'RedFRedOnGreen Manders',
                'RedFRedOnGreen Manders (Costes)',
                'GreenOnRedFRed Manders',
                'GreenOnRedFRed Manders (Costes)',

                ]

        data = []

        progress = self.display_progress()
        for i in range(n_files):
            fn = self.filenames[i]
            fn_short = fn.replace(self.directory,"")[1:]

            tif = tifffile.TiffFile(fn)
            arr = tif.asarray()
            orig_size = arr.shape[1]

            arr,was_sanitized = sanitize_stack(arr)
            _,ny,nx = arr.shape

            #extent = lc.get_extent(tif)
            tif.close()
            arr_filtered = get_filtered(arr,threshold=self.threshold)

            #Reading the region of interest
            fn_rois = make_fn_rois(fn)
            if fn_rois is not None:
                roi = read_roi.read_roi_zip(fn_rois)
                expand_roi(roi,orig_size=orig_size)
            else:
                roi = {}

            #Adding the image as a roi
            image_x = [0,nx,nx,0]
            image_y = [0,0,ny,ny]

            roi["Image"] = {"x":image_x,"y":image_y}
            keys = roi.keys()
            keys.sort()

            keys.pop(keys.index("Image"))
            keys = ["Image"]+keys

            #Loop through the ROIs
            for roi_name in keys[1:]:
                r = roi[roi_name]
                x = r['x']
                y = r['y']
                vertices = np.vstack([x,y]).T
                polygon_array = create_polygon([ny,nx], vertices)
                nzs = np.sum(polygon_array)

                #The super dicts starts here!
                dic = {}
                dic['Folder'] = self.directory
                dic['Filename'] = fn_short
                dic['ROI'] = roi_name
                dic['Area'] = nzs

                #Proportions... The channels of interest
                for c in [1,2,3]:
                    short_name = self.get_channel_name(c,short=True)

                    mask = np.ma.masked_where(polygon_array == False, arr_filtered[c,...])
                    prop = float(np.sum(mask > self.threshold))/nzs

                    dic[short_name+" Prop"] = prop

                if 1:
                    #Do 3 combos for the Manders coefficients...
                    pairs = [[1,2],[1,3],[2,3]]
                    for pair in pairs:
                        #if self.debug: print pair
                        fi = np.ma.masked_where(polygon_array == False, arr_filtered[pair[0],...]).filled(0)
                        si = np.ma.masked_where(polygon_array == False, arr_filtered[pair[1],...]).filled(0)
                        
                        fi_name = self.get_channel_name(pair[0],short=True)
                        si_name = self.get_channel_name(pair[1],short=True)
                        dic_coloc = costes_threshold(fi,si,first_image_name=fi_name, second_image_name=si_name)
                        #if self.debug: print dic_coloc
                        dic.update(dic_coloc)

                    #Do the last combo using Red x FRed
                    prod = arr_filtered[2,...]*arr_filtered[3,...]/255.
                    #if self.debug: print np.min(prod),np.max(prod)

                    fi = np.ma.masked_where(polygon_array == False, arr_filtered[1,...]).filled(0)
                    fi_name = self.get_channel_name(1,short=True)
                    si = np.ma.masked_where(polygon_array == False, prod).filled(0)
                    si_name = "RedFRed"
                    dic_coloc = costes_threshold(fi,si,first_image_name=fi_name, second_image_name=si_name)
                    dic.update(dic_coloc)
                data.append(dic)

            #Update the progress bar
            progress.value += 1

        df = pd.DataFrame(data,columns=columns)
        progress.value = 0
        return df

    def display_progress(self):
        n_files = len(self.filenames)
        progress = widgets.IntProgress(
            value=0,
            min=0,
            max=n_files,
            step=1,
            description='Processing:',
            bar_style='', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal'
        )
        display(progress)
        return progress

    def create_overlap_data(self):
        directory = self.directory
        if directory is None:
            return

        filenames = self.filenames
        n_files = len(filenames)

        columns = ['Folder', 'Filename', 'ROI', 'Area'] #,'Green Prop', 'Red Prop', 'FRed Prop']

        pairs = get_combos([1,2,3])
        for pair in pairs:
            col_name = self.get_col_name(pair,coloc=False)
            columns.append(col_name)

        progress = self.display_progress()
        data = []
        for i in range(n_files):
            fn = self.filenames[i]
            fn_short = fn.replace(self.directory,"")[1:]

            tif = tifffile.TiffFile(fn)
            arr = tif.asarray()
            orig_size = arr.shape[1]

            arr,was_sanitized = sanitize_stack(arr)
            _,ny,nx = arr.shape

            #extent = lc.get_extent(tif)
            tif.close()
            arr_filtered = get_filtered(arr,threshold=self.threshold)

            products = {}
            for pair in pairs:
                ret = np.ones((ny,nx),dtype=float)
                #Where are the channels pointing to in the GUI?
                pair_ = np.take(self.channels,pair)
                for p in pair_:
                    ret *= arr_filtered[p,...]/255.
                products[str(pair)] = ret*255.

            #Reading the region of interest
            fn_rois = make_fn_rois(fn)
            if fn_rois is not None:
                roi = read_roi.read_roi_zip(fn_rois)
                expand_roi(roi,orig_size=orig_size)
            else:
                roi = {}

            #Adding the image as a roi
            image_x = [0,nx,nx,0]
            image_y = [0,0,ny,ny]

            roi["Image"] = {"x":image_x,"y":image_y}
            keys = roi.keys()
            keys.sort()

            keys.pop(keys.index("Image"))
            keys = ["Image"]+keys

            #Loop through the ROIs
            for roi_name in keys:
                r = roi[roi_name]
                x = r['x']
                y = r['y']
                vertices = np.vstack([x,y]).T
                polygon_array = create_polygon([ny,nx], vertices)
                nzs = np.sum(polygon_array)

                vec = [self.directory,fn_short,roi_name,nzs]
                for pair in pairs:
                    mask = np.ma.masked_where(polygon_array == False, products[str(pair)])
                    prop = float(np.sum(mask > self.threshold))/nzs
                    vec.append(prop)
                    #plt.imshow(mask) #polygon_array)
                    #plt.suptitle(str(pair))

                data.append(vec)

            #Update the progress bar
            progress.value += 1

        df = pd.DataFrame(data,columns=columns)
        progress.value = 0
        return df


################################################################################
#
# MANDER COLOC
#
################################################################################

# https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
def create_polygon(shape,poly_verts):
    from matplotlib.path import Path

    ny,nx = shape

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)

    grid = grid.reshape((ny,nx))

    return grid

def get_filtered(arr,filtered=None,threshold=0):
    nc = arr.shape[0]
    if filtered is None:
        filtered = [
            None,
            [libatrous.get_kernel(libatrous.LIN3),3,6,threshold,255],
            [libatrous.get_kernel(libatrous.LIN3),3,6,threshold,255],
            [libatrous.get_kernel(libatrous.LIN3),3,6,threshold,255],
        ]

    ret = np.zeros(arr.shape,np.float32)

    for i in range(nc):
        img = arr[i,:,:]

        #Add filtering
        f = filtered[i]
        if f is not None:
            kernel = f[0]
            ls = f[1]
            hs = f[2]
            mi = f[3]
            ma = f[4]
            img = libatrous.get_bandpass(img,ls-1,hs,kernel,False)
            img[img < mi] = 0
            img[img > ma] = ma

        ret[i,...] = img

    return ret

def get_moc(arr,g_index,r_index,threshold=0,roi=None):

    arr_r = arr[r_index,...]
    arr_g = arr[g_index,...]

    #Apply masking if needed...
    if roi is not None:
        arr_r = np.ma.masked_where(roi == False, arr_r.astype(float))
        arr_g = np.ma.masked_where(roi == False, arr_g.astype(float))

    s = np.sum(arr_r)

    if s == 0:
        return np.nan

    wh = arr_g > threshold
    return float(np.sum(arr_r[wh]))/s

def get_moc_three(arr,g_index,r_index,f_index,threshold=0,roi=None):

    arr_r = arr[r_index,...].astype(float)
    arr_f = arr[f_index,...].astype(float)
    arr_g = arr[g_index,...].astype(float)

    #Apply masking if needed...
    if roi is not None:
        arr_r = np.ma.masked_where(roi == False, arr_r.astype(float))
        arr_f = np.ma.masked_where(roi == False, arr_f.astype(float))
        arr_g = np.ma.masked_where(roi == False, arr_g.astype(float))

    arr_comb = arr_r*arr_f/256.
    s = np.sum(arr_comb)

    if s == 0:
        return np.nan

    wh = arr_g > threshold
    return np.sum(arr_comb[wh])/s

#
# TIF FILES
#
def get_extent(tif):
    w = tif.pages[0].tags['ImageWidth'].value
    h = tif.pages[0].tags['ImageLength'].value

    s, ss = tif.pages[0].tags['XResolution'].value
    px_um = float(ss) / float(s)
    s, ss = tif.pages[0].tags['YResolution'].value
    py_um = float(ss) / float(s)
    extent = [0,px_um*w,0,py_um*h]
    return extent

#
# SQLITE functions
#
def find_relationship(sc,parent,child,prefix="MyExpt_"):
    query = '''SELECT relationship_type_id FROM %(prefix)sPer_RelationshipTypes
        WHERE object_name1=\'%(parent)s\' AND
            object_name2=\'%(child)s\'
    ''' % {"prefix":prefix,"parent":parent,"child":child}

    b = sc.execute(query, asarray=True)
    if len(b) == 0:
        ret = -1
    else:
        ret = b[0][0]
    return ret

def get_measurement_names(sc, table="MyExpt_Per_Image", asarray=False):
    query = '''SELECT * FROM %(table)s LIMIT 1''' % {'table':table}

    a = sc.execute(query, asarray=asarray)
    if asarray:
        ret = a.dtype.names
    else: ret = list(map(lambda x: x[0], a.description))
    return ret

 #['Green_Objects_Correlation_Overlap_Green_Filtered_Red_Filtered','Red_Objects_Correlation_Overlap_Green_Filtered_Red_Filtered'] ):
def get_data(sc,table="MyExpt_Per_Image", index="ImageNumber"):
    query = '''SELECT * FROM %(table)s''' % {'table':table}

    a = sc.execute(query)
    df = pd.DataFrame(a.fetchall())
    df.columns = list(map(lambda x: x[0], a.description))

    if index is not None:
        df.set_index(index, inplace=True)

    return df

def image_iter(sc):
    query = '''SELECT ImageNumber,Image_PathName_Green,Image_FileName_Green FROM MyExpt_Per_Image ORDER BY ImageNumber'''
    a = sc.execute(query, asarray=True)

    f = a['image_filename_green']
    p = a['image_pathname_green']
    inum = a['imagenumber']
    n = len(inum)
    ret = []
    for i in range(n):
        pp = p[i].split("/")
        ret.append([int(inum[i]),os.path.normpath("/".join(p[i].split("/")[-1:])+"/"+f[i])])

    return ret


#
# OS functions
#
def get_files(directory,match="*.tif",sanitized=False):
    ret = []

    for root, dirnames, filenames in os.walk(directory):
        for fn in fnmatch.filter(filenames, match):
            if "montage" in fn.lower():
                continue

            if fn.startswith("sanitized") and sanitized == False:
                continue

            if not fn.startswith("sanitized") and sanitized == True:
                continue

            ret.append(os.path.join(root,fn))

    ret.sort()
    return ret

#
# SANITIZATION
#
#from skimage.transform import resize
def sanitize_stack(arr,output=[1024,1024],interp='nearest'):
    was_sanitized = False
    ret = arr

    if arr.shape[0] != 4 or arr.shape[1] != 1024 or arr.shape[2] != 1024:
        was_sanitized = True
        ret = np.zeros((4,1024,1024),dtype=np.float32)

        for c in range(4):
            if c >= arr.shape[0]:
                break
            a = arr[c,...]
            ret[c,...] = imresize(a, output, interp=interp)

    return ret,was_sanitized

def sanitize(fn,arr,c1234=[0,1,2,1], output=[1024,1024],interp='nearest'):
    d,f = os.path.split(fn)

    if ".lif - Image" in f:
        a,b = f.split(".lif - Image00")
        f = a+chr(96+int(b.split(".")[0]))+".tif"

    fn = os.path.join(d,'sanitized_'+f)
    ret = []
    for c in c1234:
        if c is None:
            a = np.zeros(output,dtype=arr.dtype)
        else:
            a = arr[c,...]
            if a.shape[0] != output[0] or a.shape[1] != output[1]:
                a =imresize(a, output, interp=interp).astype(a.dtype)
                #print np.min(a),np.max(a),np.min(arr[c,...]),np.max(arr[c,...])
        ret.append(a)

    tifffile.imsave(fn,np.array(ret),imagej=True)

def sanitize_folder(directory,c1234=[0,1,2,1], output=[1024,1024],interp='nearest'):
    print("Folder: %s" % directory)
    fns = get_files(directory)
    n_fns = len(fns)
    for i in range(n_fns):
        fn=fns[i]
        _,name = directory.rsplit("\\",1)
        name+="/"+fn
        print("  %d/%d - sanitizing %s" % (i+1,n_fns,name))
    
        fn = os.path.join(directory,fn) 
        tif = tifffile.TiffFile(fn)
        arr = tif.asarray()
        if len(c1234) > arr.shape[0]:
                c1234 = [0,1,2,1]

        sanitize(fn,arr,c1234=c1234,output=output,interp=interp)
        #lc.display_channels(arr,name,layout=(2,2),figsize=(16,14))
        tif.close()

################################################################################
#
# DISPLAY FUNCTIONS
#
################################################################################

def array_to_rgb(arr,channels_rgb=[[0,0,1],[0,1,0],[1,0,0],[0,0.5,1]]):
    channels_rgb = np.array(channels_rgb,float)
    n = arr.shape[0]
    arr_temp = np.zeros([arr.shape[1],arr.shape[2],3],np.uint8)
    for i in range(n):
        for c in range(3):
            img = np.max([arr_temp[:,:,c],arr[i,:,:]*channels_rgb[i,c]],axis=0)
            img[img > 255] = 255
            img[img < 0] = 0
            arr_temp[:,:,c] = img
    return arr_temp

def add_sizebar(ax, size,color='w'):
    fontprops = fm.FontProperties(size=18)
    asb = AnchoredSizeBar(ax.transData,
                          size,
                          str(size)+"$\mu$m",
                          loc=3,
                          color=color,
                          pad=0., borderpad=0.2, sep=5, size_vertical=0.2,
                          frameon=False, fontproperties=fontprops)
        
    ax.add_artist(asb)

def add_text(ax,s,color):
    at = AnchoredText(s,
                      loc=2,
                      pad=0, borderpad=0.25,
                      frameon=False, prop=dict(size=18,color=color))

    ax.add_artist(at)

def add_circle(ax,color):
    from matplotlib.patches import Circle
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea

    ada = AnchoredDrawingArea(20, 20, -10, 3,
                              loc='upper left', pad=0, frameon=False)
    color = np.array(color)/np.max(color)
    p = Circle((10, 10), 4, color=color, edgecolor=None)
    ada.da.add_artist(p)
    ax.add_artist(ada)

def add_colorbar(ax,vmin=0,vmax=255,cmap='plasma'):
    if hasattr(ax,'cbax'):
        ax.cbax.clear()
        cbax = ax.cbax
    else:
        cbax = inset_axes(ax, width="20%", height="2%", loc=4, borderpad=4.1)
        ax.cbax = cbax

    cbax.tick_params(labelsize=15,labelcolor='k',color='k',length=0)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(cbax, ticks=[vmin,vmax], cmap=cmap, norm=norm, orientation = 'horizontal')
    cb.outline.set_visible(False)

def add_roi(ax,roi,color='k',add_name=True):
    if type(roi) != list:
        roi = [roi]

    for r in roi:
        x = r['x']
        y = r['y']

        pgon = plt.Polygon(np.vstack([x,y]).T, color=color, fill=False)
        ax.add_patch(pgon)

        if add_name:
            name = r['name']
            mx = np.average(x)
            my = np.average(y)
            ax.text(mx, my, name, horizontalalignment='center', verticalalignment='center',color=color)
            #, transform=ax.transAxes)

def display_image(ax,arr,extent,cmap=None,text_color='w',name="",size=True,color=None):

    ax.imshow(arr,extent=extent,cmap=cmap,origin='lower')
    if size == True:
        add_sizebar(ax,5,text_color)
    if cmap is not None:
        add_colorbar(ax,vmin=np.min(arr),vmax=np.max(arr),cmap=cmap)
    if color is not None:
        add_circle(ax,color)
    if name is not None:
        add_text(ax,name,text_color)

def overlay_coloc(arr,
        axes,
        channels,
        channels_rgb,
        overlap_color = [1,1,0],
        threshold = 0,
        alpha=0.2):

    assert(len(channels) == 2)
    c0 = channels[0]
    c1 = channels[1]
    cmap0 = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', [[0,0,0,0],channels_rgb[c0]+[alpha]], 1)
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', [[0,0,0,0],channels_rgb[c1]+[alpha]], 1)
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', [[0,0,0,0],overlap_color+[alpha]], 1)
    
    im_c0 = arr[c0,:,:] > threshold
    im_c1 = arr[c1,:,:] > threshold
    im_inter= np.bitwise_and(im_c0,im_c1)

    im_inter = np.ma.masked_where(im_inter == False,im_inter)
    axes[0].imshow(im_inter,cmap=cmap2,vmin=0,vmax=1,origin='lower')
    axes[1].imshow(im_inter,cmap=cmap2,vmin=0,vmax=1,origin='lower')

    im_inter = np.ma.masked_where(im_inter == False,im_inter)
    axes[0].imshow(im_inter,cmap=cmap2,vmin=0,vmax=1,origin='lower')
    axes[1].imshow(im_inter,cmap=cmap2,vmin=0,vmax=1,origin='lower')

    im_l0 = np.bitwise_and(im_c0,np.bitwise_not(im_c1))
    im_l0 = np.ma.masked_where(im_l0 == False,im_l0)
    axes[1].imshow(im_l0,cmap=cmap0,vmin=0,vmax=1,origin='lower')

    im_l1 = np.bitwise_and(im_c1,np.bitwise_not(im_c0))
    im_l1 = np.ma.masked_where(im_l1 == False,im_l1)
    axes[0].imshow(im_l1,cmap=cmap1,vmin=0,vmax=1,origin='lower')



def product(arr,channels):
    shape = arr[channels[0],...].shape
    ret = None

    for i in range(len(channels)):
        c = channels[i]
        arr_temp = arr[c,...]

        if ret is None:
            ret = arr_temp.copy()
        else:
            ret *= arr_temp.copy()
            ret /= 255.

    return ret

def make_composite(arr,axes,extent,
        channels=[0,1,3],
        composite_channels=None,
        spots = None, filtered=None,
        channel_names=['Blue','Green','Red','FarRed','Combined'],
        composite_name="Composite",
        channels_rgb=[[0,0,1],[0,1,0],[1,0,0],[0,0.5,1]],
        overlap=False, threshold=0,
    ):

    #The composite channels may be different from the display channels...
    #This allows to display (in b&w) R and G but then as the composite, display with nuclei.
    #if composite_channels is None:
    #    composite_channels = channels

    n_axes = len(axes)
    if extent is not None:
        size = True
    else:
        extent = [0,arr.shape[2],0,arr.shape[1]]
        size = False

    for i in range(n_axes):
        ax = axes[i]
        
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4][::-1])

        ax.margins(0)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axis('off')

        if i == n_axes-1:
            break
            
        #here c is the actual channel we want to display
        c = channels[i]
        arr_temp = arr[c,...]

        #Adding a circle if needed:
        circle_col = None
        if composite_channels is not None:
            circle_col = channels_rgb[c]

        display_image(ax,arr_temp,extent,cmap='gray_r',text_color='black',name="C%d - %s" % (c+1,channel_names[c]),color=circle_col,size=size)

        if overlap:
            cmap0 = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', [[0,0,0,0],channels_rgb[c]+[0.5]], 1)

            nz = arr_temp > threshold
            im_coloc = np.ma.masked_where(nz == False,arr_temp)
            ax.imshow(im_coloc,cmap=cmap0,vmin=0,vmax=1,origin='lower')

    if composite_channels is not None:
        ax = axes[-1]
        arr_temp = array_to_rgb(arr[composite_channels,...],
            np.take(channels_rgb,composite_channels,axis=0).tolist()
        )

        display_image(ax,arr_temp,extent,name=composite_name,size=size)

def rescale(arr):
    return (255.*arr.astype(np.float)/np.max(arr)).astype(np.uint8)
    
def get_subset(df_r,image_number,parent="A",child="B"):
    truth_table = (df_r["image_number1"] == image_number) & \
        (df_r["relationship"] == 'Parent') & \
        (df_r["object_name1"] == parent) & \
        (df_r["object_name2"] == child)
    subset = df_r.loc[truth_table, "object_number1"].values
    return subset

def get_subset_values(df_o,image_number,subset=None,
        variables=['Red_Objects_Location_Center_X','Red_Objects_Location_Center_Y']):
    df_i = df_o.loc[df_o["ImageNumber"]==image_number]
    if subset is None:
        ret = df_i.loc[:,variables].dropna(axis=0)
    else:
        ret = df_i.loc[subset,variables]
    return ret

def display_channels(arr,subplots,origin='upper',
        names=['Blue','Green','Red','Green 2'],RGB_name="Composite", spots = None, filtered=None):

    #Array is of the type nchannel,ny,nx
    nc = len(names) #arr.shape[0]
    if spots is None:
        spots = [None]*(nc+1)
    if filtered is None:
        filtered = [None]*(nc)
    
    for i in range(nc+1):
        ax = subplots[i]
        if i < nc:
            name = names[i]
            title = "C%d" % i
            title = title+ ' - ' + name
            img = arr[i,:,:]

            #Add filtering
            f = filtered[i]
            if f is not None:
                kernel = f[0]
                ls = f[1]
                hs = f[2]
                mi = f[3]
                ma = f[4]
                img = libatrous.get_bandpass(img,ls-1,hs,kernel,False)
                img[img < mi] = mi
                img[img > ma] = ma

            cmap='gray_r'
            im = ax.imshow(img,origin=origin,cmap=cmap) #,vmin=0,vmax=255)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im,cax=cax)
        else:
            name = "yellow"
            title = RGB_name
            rgbArray = np.zeros((arr.shape[1],arr.shape[2],3), 'uint8')
            for c in range(3):
                rgbArray[..., c] = arr[2-c,...] #rescale(arr[2-c,...])
            img = Image.fromarray(rgbArray)
            cmap = None
            ax.imshow(img,origin=origin,cmap=cmap)

        if spots[i] is not None:
            sp = spots[i]
            ax.scatter(sp[:,0],sp[:,1],s=4,marker='o',color='none', edgecolors=name.lower(), linewidth=0.5)

        ax.set_axis_off()
        ax.set_title(title)
        
def make_name(fn,folder=False):
    d,name = os.path.split(fn)
    name = name.replace("sanitized_","").replace(".tif","")
    if folder:
        dd,d = os.path.split(d)
        name = "/".join([d,name])
    return name

#Highlight by image number or object number...
def make_names(df,names_from="DAPI"):
    names = [make_name(fn) for fn in df['Image_FileName_'+names_from]]
    return names

# https://stackoverflow.com/questions/13312820/how-do-i-plot-just-the-positive-error-bar-with-pyplot-bar
def autolabel(ax, ind, values,fontsize=12):
    inv = ax.transData.inverted()
    offset = inv.transform((0,fontsize+2))[1]-inv.transform((0,0))[1]

    v = values[0]
    if isinstance(v, np.float64) or isinstance(v, np.float32) or isinstance(v, float):
        isFloat = True
    else:
        isFloat = False

    for ii,value in enumerate(values):
        height = values[ii]
        if height-offset > 0:
            if isFloat:
                s = '%.2f'
            else:
                s = '%d'
            ax.text(ind[ii], height-offset, s% values[ii], ha='center', va='bottom',
                    fontsize=fontsize,color='w',alpha=0.7)

# https://stackoverflow.com/questions/24988448/how-to-draw-vertical-lines-on-a-given-plot-in-matplotlib
def axvlines(xs, ax=None, **plot_kwargs):
    """
    Draw vertical lines on plot
    :param xs: A scalar, list, or 1D array of horizontal offsets
    :param ax: The axis (or none to use gca)
    :param plot_kwargs: Keyword arguments to be passed to plot
    :return: The plot object corresponding to the lines.
    """
    if ax is None:
        ax = plt.gca()
    xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
    lims = ax.get_ylim()
    x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
    y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
    plot = ax.plot(x_points, y_points, scaley = False, **plot_kwargs)
    return plot

def display_onevar(ax,df,var,sort=True,names_from="DAPI",index=None,col='b',coli='r',split=False,hist=False):
    n = len(df)

    if type(names_from) == str:
        names = make_names(df,names_from)
    else:
        names = names_from

    #index can be None with ==, that's fine.
    col = np.where(df.index==index, coli, col)

    values = df[var].values

    if hist == True:
        n, bins, patches = ax.hist(values, 5, range=(0,1), density=True, facecolor='g', alpha=0.75)
        v = values[np.where(df.index == index)[0][0]]
        axvlines(v,ax,color=coli,label="%.2f" % v)
        ax.set_xlim(0,1)
        ax.legend()
        
    else:
        if sort == True:
            indexes = np.argsort(values)
            values = np.take(values,indexes)
            col = col[indexes]
            names = np.array(names)[indexes]

        indexes = np.arange(n,dtype=int)
        if "Mean" in var and var.replace("Mean","StDev") in df.keys():
            var_std = var.replace("Mean","StDev")
            vstd = np.df[var_std].values
            if sort == True: vstd = vstd[indexes]
            yerr = [np.zeros(n),vstd]
        else:
            yerr = None
            #yerr = [np.zeros(n),np.zeros(n)+20]

        ax.bar(indexes,values,color=col,edgecolor='none',yerr=yerr)
        ax.set_xticks(indexes)
        ax.set_xticklabels(names)
        #ax.set_ylabel('Value for Image', fontsize=12)
        #ax.set_xlabel('ImageFile', fontsize=12)
        if split:
            var = split_var(var)

        autolabel(ax,indexes,values,fontsize=6) 

    ax.set_title(var)
    
def get_sort(v1,v2,sort='sum'):
    if sort == 'sum':
        indexes = np.argsort(v1+v2)
    elif sort == 'ratio':
        indexes = np.argsort(v1/v2)
    elif sort == 'v1':
        indexes = np.argsort(v1)
    elif sort == 'v2':
        indexes = np.argsort(v2)
    else:
        indexes = None
    return indexes

def get_sort_name(sort,v1,v2):
    if sort == 'sum':
        name = "Ordered by %s+%s" % (v1,v2)
    elif sort == 'ratio':
        name = "Ordered by %s/%s" % (v1,v2)
    elif sort == 'v1':
        name = "Ordered by %s" % v1
    elif sort == 'v2':
        name = "Ordered by %s" % v2
    else:
        name = ""
    return name

def display_twovar_bars(ax,df,var1,var2,sort='sum',names_from="DAPI",index=None,col='b',col1='r',col2='g',split=False):
    n = len(df)

    if type(names_from) == str:
        names = make_names(df,names_from)
    else:
        names = names_from

    #index can be None with ==, that's fine.
    col1_ = np.where(df.index==index, col1, col)
    col2_ = np.where(df.index==index, col2, col)

    v1 = df[var1].values
    v2 = df[var2].values

    if sort is not None:
        indexes = get_sort(v1,v2,sort)
        v1 = np.take(v1,indexes)
        v2 = np.take(v2,indexes)
        col1_ = col1_[indexes]
        col2_ = col2_[indexes]
        names = np.array(names)[indexes]

    indexes = np.arange(n,dtype=int)
    if "Mean" in var1 and var1.replace("Mean","StDev") in df.keys():
        var1_std = var1.replace("Mean","StDev")
        v1std = df[var1_std].values
        var2_std = var2.replace("Mean","StDev")
        v2std = df[var2_std].values
        if sort is not None:
            v1std = v1std[indexes]
            v2std = v2std[indexes]
        y1err = [np.zeros(n),v1std]
        y2err = [np.zeros(n),v2std]
    else:
        y1err = None
        y2err = None
        #yerr = [np.zeros(n),np.zeros(n)+20]

    width = 0.35
    p1 = ax.bar(indexes,v1,color=col1_,edgecolor='k',yerr=y1err,width=width,label=var1)
    p2 = ax.bar(indexes+width,v2,color=col2_,edgecolor='k',yerr=y2err,width=width,label=var2)
    ax.set_xticks(indexes)
    ax.set_xticklabels(names)

    #The legend
    custom_lines = [Line2D([0], [0], color=col1, lw=6),
                Line2D([0], [0], color=col2, lw=6)]
    ax.legend(custom_lines, [var1, var2], loc='upper left',frameon=False)

    #ax.set_ylabel('Value for Image', fontsize=12)
    #ax.set_xlabel('ImageFile', fontsize=12)
    var = var1+" / "+var2
    if sort is not None:
        var = var +" ( "+ get_sort_name(sort,var1,var2)+ " )"
    if split:
        var = split_var(var)
    ax.set_title(var)

    #autolabel(ax,indexes,values,fontsize=6) 
    #autolabel(ax,indexes,values,fontsize=6) 
    
def split_var(variable,length=30):
    l = variable.split('_')
    ret = ''
    tmp = ''
    while len(l) > 0:
        v = l.pop(0)
        if len(tmp+v) > length:
            ret = ret + tmp + "\n"
            tmp = ''
        tmp += v+'_'

    if tmp != '': ret = ret+tmp[:-1]
    return ret

def display_twovars(ax,df,var1,var2,names_from="DAPI", index=None,col='b',coli='r',offset=12):
    n = len(df)

    #index can be None with ==, that's fine.
    col = np.where(df.index==index, coli, col)

    if type(names_from) == str:
        names = make_names(df,names_from)
    else:
        names = names_from

    x = df[var1].values
    y = df[var2].values
    ax.scatter(x, y, marker='x', color=col)

    inv = ax.transData.inverted()
    offset = inv.transform((offset,offset))-inv.transform((0,0))

    for i in range(len(df)):
        name = names[i]
        ax.text(x[i]+offset[0], y[i]+offset[1], name, fontsize=9)

    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title("%s vs %s" % (var1,var2))

def display_page_channels(fig,df_i,df_o,df_r,index,names_from="DAPI",names=['Blue','Green','Red'],filtered=None):
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    p = df_i.loc[index]['Image_PathName_'+names_from]
    fn = df_i.loc[index]['Image_FileName_'+names_from]
    fn = os.path.join(p,fn)
    name = make_name(fn,folder=True)
    fig.suptitle(name)

    if not os.path.exists(fn):
        return

    n_rows = ((len(names)+1)/2)*2+1
    n_cols = 4
    gs = gridspec.GridSpec(n_rows,n_cols,fig,wspace=0.2,hspace=0.3, top=0.95, bottom=0.05,left=0.05, right=0.95)

    subplots = []
    spots = [None]*4
    spots[1] = get_subset_values(df_o,index, variables=['Green_Objects_Location_Center_X', 'Green_Objects_Location_Center_Y']).values
    spots[2] = get_subset_values(df_o,index, variables=['Red_Objects_Location_Center_X', 'Red_Objects_Location_Center_Y']).values
    subset = get_subset(df_r,index,parent="Red_Expanded",child="Green_Expanded")
    spots[3] = get_subset_values(df_o,index,subset,variables=[
        'Red_Objects_Location_Center_X','Red_Objects_Location_Center_Y']).values

    i = 0
    for y in range(0,n_rows-1,2):
        for x in range(0,n_cols,2):
            #print i,x,y,n_rows,n_cols
            g = gs[y:y+2,x:x+2]
            ax = fig.add_subplot(g)
            subplots.append(ax)
            i+=1
        if i == len(names):
            break

    tif = tifffile.TiffFile(fn)
    arr = tif.asarray()
    tif.close()
    #We're displaying some images...
    display_channels(arr,subplots[:4],names=['Blue','Green','Red'],spots=spots,filtered=filtered)

    #Display the number of objects...
    g = gs[-1,:]
    ax = fig.add_subplot(g)
    meas1, meas2 = ['Image_Count_Red_Objects', 'Image_Count_Green_Objects']
    display_twovar_bars(ax,df_i,meas1,meas2,sort=None,index=index,split=False)

# Index based, display a page full of variables with
def display_page(fig,df_i,df_o,df_r,index,measurements,display_image=True,names_from="DAPI",split=False):
    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    n = len(measurements)
    n_rows = n #int(np.ceil(n/4.))

    p = df_i.loc[index]['Image_PathName_'+names_from]
    fn = df_i.loc[index]['Image_FileName_'+names_from]
    fn = os.path.join(p,fn)
    name = make_name(fn,folder=True)
    fig.suptitle(name)

    if not os.path.exists(fn): display_image = False
    if display_image: n_rows += 1

    gs = gridspec.GridSpec(n_rows,4,fig,wspace=0.2,hspace=0.3, top=0.95, bottom=0.05,left=0.05, right=0.95)

    if display_image:
        subplots = []
        spots = [None]*4
        spots[1] = get_subset_values(df_o,index, variables=['Green_Objects_Location_Center_X', 'Green_Objects_Location_Center_Y']).values
        spots[2] = get_subset_values(df_o,index, variables=['Red_Objects_Location_Center_X', 'Red_Objects_Location_Center_Y']).values
        subset = get_subset(df_r,index,parent="Red_Expanded",child="Green_Expanded")
        spots[3] = get_subset_values(df_o,index,subset,variables=[
            'Red_Objects_Location_Center_X','Red_Objects_Location_Center_Y']).values

        for y in range(1):
            for x in range(4):
                g = gs[y,x]
                ax = fig.add_subplot(g)
                subplots.append(ax)

        tif = tifffile.TiffFile(fn)
        arr = tif.asarray()
        tif.close()
        #We're displaying some images...
        display_channels(arr,subplots[:4],names=['Blue','Green','Red'],spots=spots)
        subplots = subplots[4:]

    for i in range(0,n):
        meas = measurements[i]
        ax = fig.add_subplot(gs[i+display_image, :])
        if type(meas) == list:
            meas1 = meas[0]
            meas2 = meas[1]
            display_twovar_bars(ax,df_i,meas1,meas2,sort=None,index=index,split=split)
        else:
            display_onevar(ax,df_i,meas,sort=True,index=index,split=split,hist=True)


def expand_roi(roi,orig_size=512,dest_size=1024):
    for k in roi.keys():
        x = np.array(roi[k]['x'])
        y = np.array(roi[k]['y'])
        x = np.clip(x*float(dest_size)/orig_size, 0, dest_size-1)
        y = np.clip(y*float(dest_size)/orig_size, 0, dest_size-1)
        roi[k]['x'] = x
        roi[k]['y'] = y

def generate_sanity(filenames,
        filename = 'sanity_check.pdf',
        channels_rgb1 = [[0,1,4],[0,1,0],[1,0,0],[1,0,1],[1,0,0]],
        channels_rgb2 = [[0,1,4],[0,1,0],[0,1,1],[1,0,1],[0,1,1]],
        view_filtered=False,dirs=None):

    pdf_pages = PdfPages(filename)
    n_fns = len(filenames)

    for j in range(n_fns):
        fn = filenames[j]
        if "_c" in fn or "_filtered" in fn:
            continue

        tif = tifffile.TiffFile(fn)
        arr = tif.asarray()
        extent = None
        #extent = get_extent(tif)
        tif.close()
        #print fn,arr.shape
        arr_filtered = get_filtered(arr)

        combo = np.round(arr_filtered[2,:,:]*arr_filtered[3,:,:]/256.)
        arr_filtered = np.vstack([arr_filtered,[combo]])

        fn_rois = fn.replace(".tif",".zip")

        if dirs is not None:
            directory = dirs[0]
            directory_rois = dirs[1]
            fn_rois = fn_rois.replace(directory,directory_rois).replace("sanitized_","").replace(".zip",".lif.zip")
            print fn_rois

        if os.path.exists(fn_rois):
            roi = read_roi.read_roi_zip(fn_rois)
            if dirs is not None:
                expand_roi(roi,512,1024)
        else:
            roi = None
            print "No ROI defined..."
            continue

        title = " / ".join(fn.rsplit(os.sep,2)[1:]).replace("sanitized_","")

        fig, axes = plt.subplots(4, 3, figsize=(12,16))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0, hspace=0)
        axes = axes.flatten()

        make_composite(arr,axes[0:3],None,channels=[1,2],composite_channels=[0,1,2],channels_rgb=channels_rgb1)
        overlay_coloc(arr_filtered,axes[0:3],channels=[1,2],channels_rgb=channels_rgb1)
        make_composite(arr,axes[3:6],None,channels=[1,3],composite_channels=[0,1,3],channels_rgb=channels_rgb1)
        overlay_coloc(arr_filtered,axes[3:6],channels=[1,3],channels_rgb=channels_rgb1)
        make_composite(arr,axes[6:9],None,channels=[2,3],composite_channels=[0,3,2],channels_rgb=channels_rgb2)
        overlay_coloc(arr_filtered,axes[6:9],channels=[2,3],channels_rgb=channels_rgb2)
        make_composite(arr_filtered,axes[9:12],None,channels=[1,4],composite_channels=[1,4],channels_rgb=channels_rgb1)
        overlay_coloc(arr_filtered,axes[9:12],channels=[1,4],channels_rgb=channels_rgb1)

        if roi is not None:
            for i in range(12):
                c = 'k'
                if (i+1) % 3 == 0:
                    c = 'w'
                add_roi(axes[i],roi.values(),color=c)

        fig.suptitle(title,y=0.965)

        pdf_pages.savefig(fig)
        plt.close()
        #if j > 1:
        #    break

    pdf_pages.close()
    print "done!"

def generate_sanity_3chan(filenames,
        filename = 'sanity_check_3chan.pdf',
        channels_rgb3 = [[0,1,4],[0,1,0],[1,0,0],[0,0,1]],
        view_filtered=False,dirs=None):

    #This one makes two grids per image
    # First is RGB combined coloc
    # Second is R*G*B

    pdf_pages = PdfPages(filename)
    n_fns = len(filenames)

    for j in range(n_fns):
        fn = filenames[j]
        if "_c" in fn or "_filtered" in fn:
            continue

        tif = tifffile.TiffFile(fn)
        arr = tif.asarray()
        extent = None
        #extent = get_extent(tif)
        tif.close()
        #print fn,arr.shape
        arr_filtered = get_filtered(arr)

        #No combo here
        #combo = np.round(arr_filtered[2,:,:]*arr_filtered[3,:,:]/256.)
        #arr_filtered = np.vstack([arr_filtered,[combo]])

        fn_rois = fn.replace(".tif",".zip")

        if dirs is not None:
            directory = dirs[0]
            directory_rois = dirs[1]
            fn_rois = fn_rois.replace(directory,directory_rois).replace("sanitized_","").replace(".zip",".lif.zip")
            print fn_rois

        if os.path.exists(fn_rois):
            roi = read_roi.read_roi_zip(fn_rois)
            if dirs is not None:
                expand_roi(roi,512,1024)
        else:
            roi = None
            print "No ROI defined..."
            continue

        title = " / ".join(fn.rsplit(os.sep,2)[1:]).replace("sanitized_","")

        #Doing the RGB coloc 2x2
        fig, axes = plt.subplots(2, 2, figsize=(12,12))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0, hspace=0)
        axes = axes.flatten()

        channels = [1,2,3]

        make_composite(arr_filtered,axes[0:4],None,channels=channels,composite_channels=[1,2,3],channels_rgb=channels_rgb3)

        if roi is not None:
            for i in range(4):
                c = 'k'
                if i == 3:
                    c = 'w'
                add_roi(axes[i],roi.values(),color=c)

        fig.suptitle(title,y=0.965)

        pdf_pages.savefig(fig)
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(12,12))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=-0, hspace=0)
        axes = axes.flatten()

        make_composite(arr_filtered,axes[0:4],None,channels=channels,composite_channels=None,channels_rgb=channels_rgb3)

        prod = product(arr_filtered,[1,2,3])
        name = ''
        for c in channels:
            name += 'C%d x' % (c+1)
        name = name[:-2]

        display_image(axes[-1],prod,None,cmap='gray_r',text_color='black',name=name,color=None,size=False)

        if roi is not None:
            for i in range(4):
                c = 'k'
                add_roi(axes[i],roi.values(),color=c)

        fig.suptitle(title,y=0.965)
        pdf_pages.savefig(fig)
        plt.close()

        #if j > 1:
        #    break

    pdf_pages.close()
    print "done!"

def costes_threshold(fi,si,threshold=0, first_image_name = "Green", second_image_name = "Red", max_val=255.):
    # The code below was adapted from the CellProfiler source code cellprofiler/modules/measurecolocalization.py
    # which was released under the BSD license, and is licensed accordingly
    #
    # MeasureCorrelationx - extension to the MeasureCorrelation module available in CellProfiler to include
    # Overlap coefficient, Mander's coefficients, Costes' automated threshold and Rank Weight Colocalization (RWC)
    # coefficients - written by Vasanth R Singan (VRSingan @ lbl.gov)

    from scipy.linalg import lstsq
    import scipy.ndimage as scind
    import scipy.stats as scistat

    result = []

    fi = fi.flatten()/max_val
    si = si.flatten()/max_val

    #
    # Find the slope as a linear regression to
    # A * i1 + B = i2
    #
    corr = np.corrcoef((fi,si))[1,0]
    coeffs = lstsq(np.array((fi,np.ones_like(fi))).transpose(),si)[0]
    slope = coeffs[0]
    result += [[first_image_name, second_image_name, "-", "Correlation",corr],
                   [first_image_name, second_image_name, "-", "Slope",slope]]
    
    # Orthogonal Regression for Costes' automated threshold
    nonZero = (fi > 0) | (si > 0)
    
    xvar = np.var(fi[nonZero],axis=0,ddof=1)
    yvar = np.var(si[nonZero],axis=0,ddof=1)

    xmean = np.mean(fi[nonZero], axis=0)
    ymean = np.mean(si[nonZero], axis=0)

    z = fi[nonZero] + si[nonZero]
    zvar = np.var(z,axis=0,ddof=1)

    covar = 0.5 * (zvar - (xvar+yvar))

    denom = 2 * covar
    num = (yvar-xvar) + np.sqrt((yvar-xvar)*(yvar-xvar)+4*(covar*covar))
    a = (num/denom)
    b = (ymean - a*xmean)
    
    i = 1
    while(i > 0.003921568627):
        Thr_fi_c = i
        Thr_si_c = (a*i)+b
        combt = (fi < Thr_fi_c) | (si < Thr_si_c)    
        costReg = scistat.pearsonr(fi[combt], si[combt])
        if(costReg[0] <= 0):
            break
        i= i-0.003921568627
    
    # Costes' thershold calculation
    combined_thresh_c = (fi > Thr_fi_c) & (si > Thr_si_c)
    fi_thresh_c = fi[combined_thresh_c]
    si_thresh_c = si[combined_thresh_c]
    tot_fi_thr_c = fi[(fi > Thr_fi_c)].sum()
    tot_si_thr_c = si[(si > Thr_si_c)].sum()

    result += [[first_image_name, second_image_name, "-", "First Costes Threshold",Thr_fi_c*max_val],
               [first_image_name, second_image_name, "-", "Second Costes Threshold",Thr_si_c*max_val]]
    
    #Threshold as percentage of maximum intensity in each channel
    thr_fi = threshold * scistat.scoreatpercentile(fi,100.0) / 100
    thr_si = threshold * scistat.scoreatpercentile(si,100.0) / 100

    combined_thresh = (fi > thr_fi) & (si > thr_si)
    fi_thresh = fi[combined_thresh]
    si_thresh = si[combined_thresh]
    tot_fi_thr = fi[(fi > thr_fi)].sum()
    tot_si_thr = si[(si > thr_si)].sum()
    
    # Mander's Coefficient 
    M1 = 0
    M2 = 0
    M1 = fi_thresh.sum() / tot_fi_thr
    M2 = si_thresh.sum() / tot_si_thr
    result += [[first_image_name, second_image_name, "-", "Mander's Coefficient",M1],
               [second_image_name, first_image_name, "-", "Mander's Coefficient",M2]]

    # RWC Coefficient
    RWC1 = 0
    RWC2 = 0
    Rank1 = np.lexsort([fi])
    Rank2 = np.lexsort([si])
    Rank1_U = np.hstack([[False],fi[Rank1[:-1]] != fi[Rank1[1:]]])
    Rank2_U = np.hstack([[False],si[Rank2[:-1]] != si[Rank2[1:]]])
    Rank1_S = np.cumsum(Rank1_U)
    Rank2_S = np.cumsum(Rank2_U)
    Rank_im1 = np.zeros(fi.shape, dtype=int)
    Rank_im2 = np.zeros(si.shape, dtype=int)
    Rank_im1[Rank1] = Rank1_S
    Rank_im2[Rank2] = Rank2_S

    R = max(Rank_im1.max(), Rank_im2.max())+1
    Di = abs(Rank_im1 - Rank_im2)
    weight = ((R-Di) * 1.0) / R
    weight_thresh = weight[combined_thresh]
    RWC1 = (fi_thresh * weight_thresh).sum() / tot_fi_thr
    RWC2 = (si_thresh * weight_thresh).sum() / tot_si_thr
    result += [[first_image_name, second_image_name, "-", "RWC Coefficient",RWC1],
               [second_image_name, first_image_name, "-", "RWC Coefficient",RWC2]]

   
    # Costes' Automated Threshold
    C1 = 0
    C2 = 0
    C1 = fi_thresh_c.sum() / tot_fi_thr_c
    C2 = si_thresh_c.sum() / tot_si_thr_c
    result += [[first_image_name, second_image_name, "-", "Mander's Coefficient (Costes)",C1],
               [second_image_name, first_image_name, "-", "Mander's Coefficient (Costes)",C2]]
   
    
    # Overlap Coefficient
    overlap = 0
    overlap = (fi_thresh * si_thresh).sum() / np.sqrt((fi_thresh**2).sum() * (si_thresh**2).sum())  
    K1 = (fi_thresh * si_thresh).sum() / (fi_thresh**2).sum()
    K2 = (fi_thresh * si_thresh).sum() / (si_thresh**2).sum()
    result += [[first_image_name, second_image_name, "-", "Overlap Coefficient",overlap]]

    ret = {}
    for l in result:
        if "Threshold" in l[3]:
            delim = "-"
            name = l[0]+delim+l[1]+ " " +l[3]
        else:
            delim = "On"
            name = l[1]+delim+l[0]+ " " +l[3]

        name = name.replace("\'s Coefficient","s")
        if "RWC" in name or "Corr" in name or "Overlap" in name or "Slope" in name:
            continue

        ret[name] = l[-1]
    return ret


