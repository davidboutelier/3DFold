from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, \
    QLabel, QPushButton, QVBoxLayout,QProgressBar,\
    QTabWidget,QHBoxLayout,QLabel,QLineEdit,QFrame,\
    QSizePolicy, QSpinBox, QDoubleSpinBox, QCheckBox, QStatusBar,\
    QFileDialog,QGroupBox,QRadioButton

import sys
import os
import json
import psutil
from datetime import datetime
from time import sleep
from pathlib import Path
from shapely.geometry import Polygon, Point, MultiPoint, LineString, MultiLineString
import geopandas as gpd
import pandas as pd
import shutil
import numpy as np
from scipy.spatial.distance import cdist

proj_dict = {}      # project dictionary - will be updated and saved to json for reload
lines = {}          # a dictionary for making of lines - will be overwritten
points = {}         # a dictionary for making of points - will be overwritten
grid = {}           # a dictionary for making the grid - will be discarded


class CMPWorker(QThread):
     finished = pyqtSignal()
     countChanged = pyqtSignal(int)
     partChanged = pyqtSignal(int)

     

     def run(self):

          part = 1
          dir = proj_dict["dir"]
          max_offset = proj_dict["fold_max_offset"] 
          receiver_points_file = proj_dict["receiver_point_file"]
          source_points_file = proj_dict["source_point_file"] 
          partitioning = proj_dict["partitioning"]

          cmp_file_exist = proj_dict["cmp_file_exist"]

          if cmp_file_exist:
               if os.path.exists(os.path.join(proj_dict["dir"], proj_dict["option"])):
                    pass
               else:
                    os.makedirs(os.path.join(proj_dict["dir"], proj_dict["option"]))
               
               shutil.unpack_archive(proj_dict["cmp_file"], os.path.join(proj_dict["dir"], proj_dict["option"]))
               all_cmp = gpd.read_file(os.path.join(proj_dict["dir"], proj_dict["option"], 'cmp','cmp.shp'))
               shutil.rmtree(os.path.join(os.path.join(proj_dict["dir"], proj_dict["option"], 'cmp')))
               
          else:
               # cmp file does not exist so we must calculate the cmp
               parts = os.path.split(receiver_points_file)
               path = parts[0]
               filename = parts[1]
               basename = os.path.splitext(filename)[0]
               extension = os.path.splitext(filename)[1]
               shutil.unpack_archive(receiver_points_file, os.path.join(proj_dict["dir"], 'temp'))
               RP_shp_file = os.path.join(proj_dict["dir"], 'temp', basename, basename+'.shp')
               RP_geopandas = gpd.read_file(RP_shp_file)
               shutil.rmtree(os.path.join(proj_dict["dir"], 'temp'))

               parts = os.path.split(source_points_file)
               path = parts[0]
               filename = parts[1]
               basename = os.path.splitext(filename)[0]
               extension = os.path.splitext(filename)[1]
               shutil.unpack_archive(source_points_file, os.path.join(proj_dict["dir"], 'temp'))
               SP_shp_file = os.path.join(proj_dict["dir"], 'temp', basename, basename+'.shp')
               SP_geopandas = gpd.read_file(SP_shp_file)
               shutil.rmtree(os.path.join(proj_dict["dir"], 'temp'))

               # convert gpds to numpy arrays 
               r = []
               for p in RP_geopandas.geometry:
                    r.append([p.x, p.y])
               nr = np.array(r)

               s = []
               for p in SP_geopandas.geometry:
                    s.append([p.x, p.y])
               ns = np.array(s).copy()

               # make a temp folder
               os.makedirs(os.path.join(proj_dict["dir"],'temp'))

               # define a chunk
               chunks = partitioning
               offset = max_offset
               chunk = int(len(s) / chunks)

               for i in range(0,partitioning):
                    percent = int(100*(i/partitioning))
                    self.countChanged.emit(percent)

                    if i ==0:
                         sub = ns[:chunk].copy()
                    elif i== chunks-1:
                         sub = ns[i * chunk :].copy()
                    else:
                         sub = ns[i * chunk :(i + 1) * chunk].copy()

                    dist = cdist(sub, nr, 'euclidean')
                    indices_S, indices_R = np.where(dist < offset)

                    s2 = np.copy(sub[indices_S, :], order='F').astype('float32')
                    r2 = np.copy(nr[indices_R, :], order='F').astype('float32') 
                    points = s2 + 0.5 * (r2 - s2)   
                    np.savetxt(os.path.join(proj_dict["dir"],'temp', 'temp_'+str(i)+'.csv'), points, delimiter=",") 


               part = 2
               self.partChanged.emit(part)

               # now we assemble the parts
               for i in range(0,partitioning):
                    percent = int(100*i/partitioning)
                    self.countChanged.emit(percent)

                    filename = os.path.join(dir,'temp', 'temp_'+str(i)+'.csv')

                    arr = np.genfromtxt (filename, delimiter=",")
                    this_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=arr[:,0], y=arr[:,1]), crs = 'EPSG:'+str(proj_dict["epsg"]))
                    del arr

                    if i ==0:
                         all_cmp = this_gdf.copy()
                    else:
                         all_cmp = pd.concat([all_cmp, this_gdf])

               shutil.rmtree(os.path.join(dir, 'temp'))


               save_cmp = proj_dict["save_cmp"]

               if save_cmp:
                    if os.path.exists(os.path.join(proj_dict["dir"], proj_dict["option"], 'cmp')):
                         pass
                    else:
                         os.makedirs(os.path.join(proj_dict["dir"], proj_dict["option"], 'cmp'))

                    all_cmp.to_file(os.path.join(proj_dict["dir"], proj_dict["option"], 'cmp', 'cmp.shp'))
                    MainWindow.make_archive(MainWindow, os.path.join(proj_dict["dir"], proj_dict["option"], 'cmp'), os.path.join(proj_dict["dir"], proj_dict["option"], 'cmp.zip'))
                    shutil.rmtree(os.path.join(proj_dict["dir"], proj_dict["option"], 'cmp'))

               else:
                    pass

               
          part = 3
          self.partChanged.emit(part)

          cmp = all_cmp
          partition = partitioning

          n_cmp = len(all_cmp)
          chunk = int(n_cmp / partitioning)

          shutil.unpack_archive(proj_dict["grid_file"], os.path.join(proj_dict["dir"], 'temp'))
          bins = gpd.read_file(os.path.join(proj_dict["dir"], 'temp', 'grd', 'grd.shp'))
          shutil.rmtree(os.path.join(proj_dict["dir"], 'temp'))
          b = bins.copy()

          for i in range(0, partitioning):
               percent = int(100*i/partitioning)
               self.countChanged.emit(percent)

               if i == 0:
                    m0 = cmp[:chunk].copy()
               elif i == partition - 1:
                    m0 = cmp[(i * chunk) :].copy()
               else:
                    m0 = cmp[(i * chunk) :(i + 1) * chunk].copy()

               reindexed = b.reset_index().rename(columns={'index': 'bins_index'})
               joined = gpd.tools.sjoin(reindexed, m0, predicate='contains')
               bin_stats = joined.groupby('bins_index').agg({'fold': len})
               arr = np.array(bin_stats.index)

               for k in range(len(arr)):
                    index = arr[k]
                    fold = int(bin_stats.iloc[k, 0])
                    bins.loc[index, 'fold'] = bins.loc[index, 'fold'] + fold
                    bins.loc[index, 'fold']

          if os.path.exists(os.path.join(proj_dict["dir"], proj_dict["option"], 'grd')):
               pass
          else:
               os.makedirs(os.path.join(proj_dict["dir"], proj_dict["option"], 'grd'))

          bins.to_file(os.path.join(proj_dict["dir"], proj_dict["option"], 'grd', 'grd.shp'))
          os.remove(proj_dict["grid_file"])
          MainWindow.make_archive(MainWindow, os.path.join(proj_dict["dir"], proj_dict["option"], 'grd'), proj_dict["grid_file"])
          shutil.rmtree(os.path.join(proj_dict["dir"], proj_dict["option"], 'grd'))
          self.finished.emit()


class GridMaker(QThread):
     finished = pyqtSignal()
     countChanged = pyqtSignal(int)

     def run(self):
          origin = proj_dict["origin"]
          inline_length = proj_dict["inline_length"]
          inline_azimuth = proj_dict["inline_azimuth"]
          crossline_length = proj_dict["crossline_length"]

          inline_bin_size = proj_dict["inline_bin_size"]
          crossline_bin_size = proj_dict["crossline_bin_size"]

          nbin_inline = int(np.ceil(inline_length/inline_bin_size))
          nbin_crossline = int(np.ceil(crossline_length/crossline_bin_size))
          nbin = nbin_inline * nbin_crossline
          k = 0

          dxxb = inline_bin_size * np.cos((90-inline_azimuth) * np.pi / 180)
          dxyb = inline_bin_size * np.sin((90-inline_azimuth) * np.pi / 180)

          dyxb = crossline_bin_size * np.cos((inline_azimuth) * np.pi / 180)
          dyyb = -1*crossline_bin_size * np.sin((inline_azimuth) * np.pi / 180)

          chunks = origin.split(',')

          xc = float(chunks[0])
          yc = float(chunks[1])

          dest = []

          for i in range(0,nbin_inline):
               for j in range(0,nbin_crossline):
                    k = k+1   # processed bins
                    #sleep(0.1)
                    a = np.array([xc + i * dxxb + j * dyxb, yc + j * dyyb + i * dxyb])
                    b = a + np.array([dxxb, dxyb]) 
                    c = b + np.array([dyxb, dyyb])
                    d = a + np.array([dyxb, dyyb])

                    dest.append(Polygon([a, b, c, d]))

                    percent = int(100 * k/nbin)
                    self.countChanged.emit(percent)
          
          grd = gpd.GeoDataFrame(geometry=dest, crs='epsg:'+str(proj_dict["epsg"]))
          grd['fold'] = 0

          if os.path.exists(os.path.join(proj_dict["dir"], proj_dict["option"], 'grd')):
               pass
          else:
               os.makedirs(os.path.join(proj_dict["dir"], proj_dict["option"], 'grd'))

          grd.to_file(os.path.join(proj_dict["dir"], proj_dict["option"], 'grd', 'grd.shp'))
          MainWindow.make_archive(MainWindow, os.path.join(proj_dict["dir"], proj_dict["option"], 'grd'), os.path.join(proj_dict["dir"], proj_dict["option"], 'grd.zip'))
          shutil.rmtree(os.path.join(proj_dict["dir"], proj_dict["option"], 'grd'))



          self.finished.emit()

class PointWorker(QThread):
     finished = pyqtSignal()
     countChanged = pyqtSignal(int)

     def run(self):
          pointtype = points["pointtype"]
          point_spacing = points["spacing"]
          linesfile = points["lines_file"]

          parts = os.path.split(linesfile)
          path = parts[0]
          filename = parts[1]
          basename = os.path.splitext(filename)[0]
          extension = os.path.splitext(filename)[1]

          shutil.unpack_archive(linesfile, os.path.join(proj_dict["dir"], 'temp'))

          lines_shp_file = os.path.join(proj_dict["dir"], 'temp', basename, basename+'.shp')
          lines_geopandas = gpd.read_file(lines_shp_file)
          num_lines = len(lines_geopandas)

          line_label = []
          point_label = []
          point_unique_ID = []
          x_points = []
          y_points = []

          for i in range(0,num_lines):
               
               self.countChanged.emit(int(100*((i-1)/num_lines)))

               this_line = lines_geopandas.iloc[i]
               this_line_FID = this_line["line"]
               this_line_geom = this_line["geometry"]

               xy = np.array(this_line_geom.coords.xy).T
               n_points = xy.shape[0]

               left = 0
               shift = 0

               point_num = 1

               for i in range(0, n_points-1):
                    if i==0:
                         x_points.append(xy[i,0])
                         y_points.append(xy[i,1])
                         line_label.append(str(this_line_FID).zfill(5))
                         point_label.append(str(point_num).zfill(5))
                         point_unique_ID.append(str(this_line_FID).zfill(5)+str(point_num).zfill(5))

                         seg_len = np.sqrt(np.power((xy[i,0] -xy[i+1,0]),2) + np.power((xy[i,1] - xy[i+1, 1]),2))
                    
                    
                    if (seg_len + left) > point_spacing:
                         seg_n_points = int(np.floor((seg_len - shift + left) / point_spacing))
                         newleft = seg_len + left -seg_n_points * point_spacing

                         for j in range(0, seg_n_points):
                              theta = np.arctan2(xy[i+1, 1] -xy[i, 1], xy[i+1, 0] -xy[i,0])
                              dx = (point_spacing + shift -left) * np.cos(theta) + j * point_spacing * np.cos(theta)
                              dy = (point_spacing + shift -left) * np.sin(theta) + j * point_spacing * np.sin(theta)
                              x_points.append(xy[i,0] + dx)
                              y_points.append(xy[i,1] + dy)
                              point_num = point_num+1
                              line_label.append(str(this_line_FID).zfill(5))
                              point_label.append(str(point_num).zfill(5))
                              point_unique_ID.append(str(this_line_FID).zfill(5)+str(point_num).zfill(5))

                         left = np.copy(newleft)
                    else:
                         left = left + seg_len     
               
               
          all_points = [Point(sx,sy) for sx, sy in zip(x_points, y_points)]  

          d = {'ID':point_unique_ID,'line':line_label, 'point':point_label, 'geometry': all_points}
          points_df = gpd.GeoDataFrame(d, crs='epsg:'+str(proj_dict["epsg"]))

          # TO DO
          # clip and mask the points


          # saving everything to a zipped shp

          if os.path.exists(os.path.join(proj_dict["dir"], proj_dict["option"], pointtype+'_points')):
               pass
          else:
               os.makedirs(os.path.join(proj_dict["dir"], proj_dict["option"], pointtype+'_points'))

          points_df.to_file(os.path.join(proj_dict["dir"], proj_dict["option"], pointtype+'_points', pointtype+'_points.shp'))
          MainWindow.make_archive(MainWindow, os.path.join(proj_dict["dir"], proj_dict["option"], pointtype+'_points'), os.path.join(proj_dict["dir"], proj_dict["option"], pointtype+'_points.zip'))
          shutil.rmtree(os.path.join(proj_dict["dir"], proj_dict["option"], pointtype+'_points'))


          if pointtype == 'source':
               proj_dict["source_point_file_exist"] = True
               proj_dict["source_point_file"] = os.path.join(proj_dict["dir"], proj_dict["option"], pointtype+'_points.zip')
          else:
               proj_dict["receiver_point_file_exist"] = True
               proj_dict["receiver_point_file"] = os.path.join(proj_dict["dir"], proj_dict["option"], pointtype+'_points.zip')

          self.finished.emit()
          

class LinesWorker(QThread):
     finished = pyqtSignal()
     countChanged = pyqtSignal(int)

     def run(self):
          global proj_dict
          count = 0
          linetype = lines["type"]
          length = lines["length"]
          width = lines["width"]
          azimuth = lines["azimuth"]
          spacing = lines["spacing"]
          inline_shift = lines["inline_shift"]
          crossline_shift = lines["crossline_shift"]
          start_number = lines["start_number"]
          inc_number = lines["inc_number"]

          proj_folder = proj_dict["dir"]
          option = proj_dict["option"]
          inline_length = proj_dict["inline_length"]
          inline_azimuth = proj_dict["inline_azimuth"]
          crossline_length = proj_dict["crossline_length"]
          epsg = int(proj_dict["epsg"])
          origin = proj_dict["origin"].split(",")
          origin_x = float(origin[0])
          origin_y = float(origin[1])

          new_lines = []
          line_ID = []

          if linetype == "source":
               num_lines = int(np.floor((width - inline_shift) / spacing))+1

               for i in range(0,num_lines):
                    xs = origin_x + (inline_shift  + i*spacing ) * np.cos((360 + 90-(azimuth-90))%360 * np.pi/180) + crossline_shift * np.cos((360+90-(azimuth))%360 * np.pi/180)
                    ys = origin_y + (inline_shift + i*spacing ) * np.sin((360 + 90-(azimuth-90))%360 * np.pi/180) + crossline_shift * np.sin((360+90-(azimuth))%360 * np.pi/180)

                    xe = xs + (length-crossline_shift) * np.cos((360+90-azimuth)%360 * np.pi/180)
                    ye = ys + (length-crossline_shift) * np.sin((360+90-azimuth)%360 * np.pi/180)

                    line = LineString([(xs,ys), (xe, ye)])
                    new_lines.append(line)
                    line_ID.append(start_number+i*inc_number)

                    self.countChanged.emit(int(100*(i/num_lines)))

               proj_dict["source_line_file_exist"] = True
               proj_dict["source_line_file"] = os.path.join(proj_folder, option, linetype+'_lines.zip') 

          else:
               num_lines = int(np.floor((width - crossline_shift) / spacing))+1

               for i in range(0,num_lines):
                    xs = origin_x + (crossline_shift  + i*spacing ) * np.cos((360 + 90-(azimuth+90))%360 * np.pi/180) + inline_shift * np.cos((360+90-(azimuth))%360 * np.pi/180)
                    ys = origin_y + (crossline_shift + i*spacing ) * np.sin((360 + 90-(azimuth+90))%360 * np.pi/180) + inline_shift * np.sin((360+90-(azimuth))%360 * np.pi/180)

                    xe = xs + (length-inline_shift) * np.cos((360+90-azimuth)%360 * np.pi/180)
                    ye = ys + (length-inline_shift) * np.sin((360+90-azimuth)%360 * np.pi/180)

                    line = LineString([(xs,ys), (xe, ye)])
                    new_lines.append(line)
                    line_ID.append(start_number+i*inc_number)

                    self.countChanged.emit(int(100*(i/num_lines)))

               proj_dict["receiver_line_file_exist"] = True
               proj_dict["receiver_line_file"] = os.path.join(proj_folder, option, linetype+'_lines.zip')
          
          d = {'line':line_ID,  'geometry': new_lines}
          lines_gpd = gpd.GeoDataFrame(d, crs='epsg:'+str(epsg))

          if os.path.exists(os.path.join(proj_folder, option, linetype+'_lines')):
               pass
          else:
               os.makedirs(os.path.join(proj_folder, option, linetype+'_lines'))

          lines_gpd.to_file(os.path.join(proj_folder, option, linetype+'_lines', linetype+'_lines.shp'))
          MainWindow.make_archive(MainWindow, os.path.join(proj_folder, option, linetype+'_lines'), os.path.join(proj_folder, option, linetype+'_lines.zip'))
          shutil.rmtree(os.path.join(proj_folder, option, linetype+'_lines'))
          self.finished.emit()


class MainWindow(QWidget):

     def __init__(self):
          super().__init__()

          user = psutil.Process().username()
          proj_dict["user"] = user

          now = datetime.now()
          dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
          proj_dict["datetime"] = dt_string

          self.tabs = QTabWidget()
          self.project_tab = QWidget()
          self.lines_tab = QWidget()
          self.points_tab = QWidget()
          self.foldmap_tab = QWidget()

          self.tabs.addTab(self.project_tab,"Project")
          self.tabs.addTab(self.lines_tab,"Lines")
          self.tabs.addTab(self.points_tab,"Points")
          self.tabs.addTab(self.foldmap_tab,"Foldmap")

          project_folder = QWidget()
          project_folder.layout = QHBoxLayout()
          project_folder_label = QLabel(" Project folder: ")
          project_folder_label.setFixedWidth(100)
          project_folder.layout.addWidget(project_folder_label)
          self.project_folder_edit = QLineEdit()
          project_folder.layout.addWidget(self.project_folder_edit)
          self.project_folder_button = QPushButton(" Set project folder ")
          self.project_folder_button.setMaximumWidth(150)
          project_folder.layout.addWidget(self.project_folder_button)
          project_folder.setLayout(project_folder.layout)

          job_number = QWidget()
          job_number.layout = QHBoxLayout()
          job_number_label = QLabel(" Job number: ")
          job_number_label.setFixedWidth(100)
          job_number.layout.addWidget(job_number_label)
          self.job_number_edit=QLineEdit()
          job_number.layout.addWidget(self.job_number_edit)
          job_number.setLayout(job_number.layout)

          client_name = QWidget()
          client_name.layout = QHBoxLayout()
          client_name_label = QLabel(" Client: ")
          client_name_label.setFixedWidth(100)
          client_name.layout.addWidget(client_name_label)
          self.client_name_edit=QLineEdit()
          client_name.layout.addWidget(self.client_name_edit)
          client_name.setLayout(client_name.layout)

          project_name = QWidget()
          project_name.layout = QHBoxLayout()
          project_name_label = QLabel(" Project: ")
          project_name_label.setFixedWidth(100)
          project_name.layout.addWidget(project_name_label)
          self.project_name_edit=QLineEdit()
          project_name.layout.addWidget(self.project_name_edit)
          project_name.setLayout(project_name.layout)

          option_name = QWidget()
          option_name.layout = QHBoxLayout()
          option_name_label = QLabel(" Option: ")
          option_name_label.setFixedWidth(100)
          option_name.layout.addWidget(option_name_label)
          self.option_name_edit=QLineEdit()
          option_name.layout.addWidget(self.option_name_edit)
          option_name.setLayout(option_name.layout)

          epsg_name = QWidget()
          epsg_name.layout = QHBoxLayout()
          epsg_name_label = QLabel(" Proj (epsg): ")
          epsg_name_label.setFixedWidth(100)
          epsg_name.layout.addWidget(epsg_name_label)
          self.epsg_name_edit=QLineEdit()
          epsg_name.layout.addWidget(self.epsg_name_edit)
          epsg_name.setLayout(epsg_name.layout)

          horizontal_line = QWidget()
          horizontal_line.layout = QHBoxLayout()
          line = QFrame()
          line.setFrameShape(QFrame.HLine)
          line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
          horizontal_line.layout.addWidget(line)
          horizontal_line.setLayout(horizontal_line.layout)

          # origin of survey grid
          survey_origin = QWidget()
          survey_origin.layout = QHBoxLayout()
          survey_origin_label = QLabel(" Origin: ")
          survey_origin_label.setFixedWidth(100)
          survey_origin.layout.addWidget(survey_origin_label)
          self.survey_origin_edit=QLineEdit()
          survey_origin.layout.addWidget(self.survey_origin_edit)
          self.survey_origin_button = QPushButton('make shp')
          survey_origin.layout.addWidget(self.survey_origin_button)
          survey_origin.setLayout(survey_origin.layout)

          survey_area = QWidget()
          survey_area.layout = QHBoxLayout()
          survey_area_inline_length_label = QLabel(" Inline length: ")
          survey_area_inline_length_label.setFixedWidth(100)
          survey_area.layout.addWidget(survey_area_inline_length_label)
          self.survey_area_inline_length_spin = QSpinBox()
          self.survey_area_inline_length_spin.setMinimumWidth(150)
          self.survey_area_inline_length_spin.setMaximum(99999999)
          self.survey_area_inline_length_spin.setMinimum(-99999999)
          survey_area.layout.addWidget(self.survey_area_inline_length_spin)
          survey_area_inline_azimuth_label = QLabel(" Inline azimuth: ")
          survey_area_inline_azimuth_label.setFixedWidth(100)
        
          survey_area.layout.addStretch()
          survey_area.layout.addWidget(survey_area_inline_azimuth_label)
          self.survey_area_inline_azimuth_spin = QDoubleSpinBox()
          self.survey_area_inline_azimuth_spin.setMinimumWidth(150)
          self.survey_area_inline_azimuth_spin.setMaximum(359.99)
          self.survey_area_inline_azimuth_spin.setMinimum(0)
          
          survey_area.layout.addWidget(self.survey_area_inline_azimuth_spin)
          survey_area_crossline_length_label = QLabel(" Crossline length: ")
          survey_area_crossline_length_label.setFixedWidth(100)
          survey_area.layout.addStretch()
          survey_area.layout.addWidget(survey_area_crossline_length_label)
          self.survey_area_crossline_length_spin = QSpinBox()
          self.survey_area_crossline_length_spin.setMinimumWidth(150)
          self.survey_area_crossline_length_spin.setMaximum(99999999)
          self.survey_area_crossline_length_spin.setMinimum(-99999999)

          survey_area.layout.addWidget(self.survey_area_crossline_length_spin)
          survey_area.layout.addStretch()
          self.survey_area_pushbutton = QPushButton(" make outline ")
          survey_area.layout.addWidget(self.survey_area_pushbutton)
          survey_area.setLayout(survey_area.layout)

          #crop_area = QWidget()
          #crop_area.layout = QHBoxLayout()
          #self.crop_area_check = QCheckBox(" crop area ")
          #self.crop_area_check.setFixedWidth(100)
          #self.crop_area_edit = QLineEdit()
          #self.crop_area_button = QPushButton(" select shp file")
          #crop_area.layout.addWidget(self.crop_area_check)
          #crop_area.layout.addWidget(self.crop_area_edit)
          #crop_area.layout.addWidget(self.crop_area_button)
          #crop_area.setLayout(crop_area.layout)

          #exclusions = QWidget()
          #exclusions.layout = QHBoxLayout()
          #self.exclusions_check = QCheckBox(" exclusions ")
          #self.exclusions_check.setFixedWidth(100)
          #self.exclusions_edit = QLineEdit()
          #self.exclusions_button = QPushButton(" select shp file")
          #exclusions.layout.addWidget(self.exclusions_check)
          #exclusions.layout.addWidget(self.exclusions_edit)
          #exclusions.layout.addWidget(self.exclusions_button)
          #exclusions.setLayout(exclusions.layout)

          parameters = QWidget()
          parameters.layout = QHBoxLayout()
          self.parameters_save_button = QPushButton(" save parameters ")
          self.parameters_load_button = QPushButton(" load parameters ")
          parameters.layout.addWidget(self.parameters_save_button)
          parameters.layout.addWidget(self.parameters_load_button)
          parameters.setLayout(parameters.layout)
        
          self.project_tab.layout = QVBoxLayout(self)
          self.project_tab.layout.addWidget(project_folder)
          self.project_tab.layout.addWidget(job_number)
          self.project_tab.layout.addWidget(client_name)
          self.project_tab.layout.addWidget(project_name)
          self.project_tab.layout.addWidget(option_name)
          self.project_tab.layout.addWidget(epsg_name)
          self.project_tab.layout.addWidget(horizontal_line)   # horizontal line
          self.project_tab.layout.addWidget(survey_origin)
          self.project_tab.layout.addWidget(survey_area)
          #self.project_tab.layout.addWidget(crop_area)
          #self.project_tab.layout.addWidget(exclusions)
          
          
          self.project_tab.layout.addStretch()
          self.project_tab.setLayout(self.project_tab.layout)

          # action from tab 1        
          self.project_folder_button.clicked.connect(self.pick_project_folder)
          self.parameters_save_button.clicked.connect(self.save_project_parameters)
          self.parameters_load_button.clicked.connect(self.load_project_parameters)
          self.survey_origin_button.clicked.connect(self.make_origin)
          self.survey_area_pushbutton.clicked.connect(self.make_outline)


          # tab 2 objects
          # Receiver line spacing
          lines_tab_2groupbox = QWidget()
          lines_tab_2groupbox.layout = QHBoxLayout()

          groupbox_receiver_lines = QGroupBox(" Receiver lines")
          groupbox_source_lines = QGroupBox(" Source lines")

          vbox_receiver_lines = QVBoxLayout()
          groupbox_receiver_lines.setLayout(vbox_receiver_lines)

          vbox_source_lines = QVBoxLayout()
          groupbox_source_lines.setLayout(vbox_source_lines)

          radiobutton_new_receiver_lines = QRadioButton(" Create new receiver lines ")
          radiobutton_new_receiver_lines.setChecked(True)

          lines_tab_receiver_lines_spacing = QWidget()
          lines_tab_receiver_lines_spacing.layout = QHBoxLayout()
          lines_tab_receiver_lines_spacing_label = QLabel(" Receiver line spacing: ")
          self.lines_tab_receiver_lines_spacing_spin = QDoubleSpinBox()
          self.lines_tab_receiver_lines_spacing_spin.setMinimumWidth(150)
          self.lines_tab_receiver_lines_spacing_spin.setMaximum(999999)
          self.lines_tab_receiver_lines_spacing_spin.setMinimum(-999999)
          lines_tab_receiver_lines_spacing.layout.addWidget(lines_tab_receiver_lines_spacing_label)
          lines_tab_receiver_lines_spacing.layout.addWidget(self.lines_tab_receiver_lines_spacing_spin)
          lines_tab_receiver_lines_spacing.setLayout(lines_tab_receiver_lines_spacing.layout)

          lines_tab_receiver_lines_shifts = QWidget()
          lines_tab_receiver_lines_shifts.layout = QHBoxLayout()
          lines_tab_receiver_lines_shifts_label = QLabel(" Shifts ")
          lines_tab_receiver_lines_shifts_label2 = QLabel(" Inline ")
          lines_tab_receiver_lines_shifts_label3 = QLabel(" Crossline ")
          self.lines_tab_receiver_lines_shifts_spin1 = QDoubleSpinBox()
          self.lines_tab_receiver_lines_shifts_spin2 = QDoubleSpinBox()

          self.lines_tab_receiver_lines_make_shp_button = QPushButton(" Make shp ")

          lines_tab_receiver_lines_shifts.layout.addWidget(lines_tab_receiver_lines_shifts_label)
          lines_tab_receiver_lines_shifts.layout.addStretch()
          lines_tab_receiver_lines_shifts.layout.addWidget(lines_tab_receiver_lines_shifts_label2)
          lines_tab_receiver_lines_shifts.layout.addWidget(self.lines_tab_receiver_lines_shifts_spin1)
          lines_tab_receiver_lines_shifts.layout.addStretch()
          lines_tab_receiver_lines_shifts.layout.addWidget(lines_tab_receiver_lines_shifts_label3)
          lines_tab_receiver_lines_shifts.layout.addWidget(self.lines_tab_receiver_lines_shifts_spin2)
          #lines_tab_receiver_lines_shifts.layout.addStretch()
          #lines_tab_receiver_lines_shifts.layout.addWidget(self.lines_tab_receiver_lines_make_shp_button)
          lines_tab_receiver_lines_shifts.setLayout(lines_tab_receiver_lines_shifts.layout)
          
          #lines_tab_receiver_lines_use_file = QWidget()
          #lines_tab_receiver_lines_use_file.layout = QHBoxLayout()
          #lines_tab_receiver_lines_use_file_label = QLabel(" receiver lines shp file: ")
          self.lines_tab_receiver_lines_use_file_edit = QLineEdit()
          self.lines_tab_receiver_lines_use_file_button = QPushButton(" Select shp file ")
          #lines_tab_receiver_lines_use_file.layout.addWidget(lines_tab_receiver_lines_use_file_label)
          #lines_tab_receiver_lines_use_file.layout.addWidget(self.lines_tab_receiver_lines_use_file_edit)
          #lines_tab_receiver_lines_use_file.layout.addWidget(self.lines_tab_receiver_lines_use_file_button)
          #lines_tab_receiver_lines_use_file.setLayout(lines_tab_receiver_lines_use_file.layout)

          self.lines_tab_receiver_use_file_radiobutton = QRadioButton(" Use lines from file")
          self.lines_tab_receiver_line_start_number = QWidget()
          self.lines_tab_receiver_line_start_number_layout = QHBoxLayout()
          self.lines_tab_receiver_line_start_number_label = QLabel(" Lines starting number ")
          self.lines_tab_receiver_line_start_number_spin = QSpinBox()
          self.lines_tab_receiver_line_start_number_spin.setMaximum(999999)
          self.lines_tab_receiver_line_start_number_spin.setMinimum(-999999)
          self.lines_tab_receiver_line_start_number_layout.addWidget(self.lines_tab_receiver_line_start_number_label)
          self.lines_tab_receiver_line_start_number_layout.addWidget(self.lines_tab_receiver_line_start_number_spin)
          self.lines_tab_receiver_line_start_number.setLayout(self.lines_tab_receiver_line_start_number_layout)

          self.lines_tab_receiver_line_inc_number = QWidget()
          self.lines_tab_receiver_line_inc_number_layout = QHBoxLayout()
          self.lines_tab_receiver_line_inc_number_label = QLabel(" Lines increment number ")
          self.lines_tab_receiver_line_inc_number_spin = QSpinBox()
          self.lines_tab_receiver_line_inc_number_spin.setMaximum(999999)
          self.lines_tab_receiver_line_inc_number_spin.setMinimum(-999999)
          self.lines_tab_receiver_line_inc_number_layout.addWidget(self.lines_tab_receiver_line_inc_number_label)
          self.lines_tab_receiver_line_inc_number_layout.addWidget(self.lines_tab_receiver_line_inc_number_spin)
          self.lines_tab_receiver_line_inc_number.setLayout(self.lines_tab_receiver_line_inc_number_layout)

          vbox_receiver_lines.addWidget(radiobutton_new_receiver_lines)
          vbox_receiver_lines.addWidget(lines_tab_receiver_lines_spacing)
          vbox_receiver_lines.addWidget(lines_tab_receiver_lines_shifts)
          vbox_receiver_lines.addWidget(self.lines_tab_receiver_line_start_number)
          vbox_receiver_lines.addWidget(self.lines_tab_receiver_line_inc_number)
          vbox_receiver_lines.addWidget(self.lines_tab_receiver_lines_make_shp_button)
          vbox_receiver_lines.addWidget(self.lines_tab_receiver_use_file_radiobutton)
          vbox_receiver_lines.addWidget(self.lines_tab_receiver_lines_use_file_button)
          vbox_receiver_lines.addWidget(self.lines_tab_receiver_lines_use_file_edit)
        
          # source lines
          radiobutton_new_source_lines = QRadioButton(" Create new source lines")
          radiobutton_new_source_lines.setChecked(True)
          
          lines_tab_source_lines_spacing = QWidget()
          lines_tab_source_lines_spacing.layout = QHBoxLayout()
          lines_tab_source_lines_spacing_label = QLabel(" Source line spacing: ")
          self.lines_tab_source_lines_spacing_spin = QDoubleSpinBox()
          self.lines_tab_source_lines_spacing_spin.setMinimumWidth(150)
          self.lines_tab_source_lines_spacing_spin.setMaximum(999999)
          self.lines_tab_source_lines_spacing_spin.setMinimum(-999999)
          self.lines_tab_source_lines_spacing_spin.setValue(50)
          lines_tab_source_lines_spacing.layout.addWidget(lines_tab_source_lines_spacing_label)
          lines_tab_source_lines_spacing.layout.addWidget(self.lines_tab_source_lines_spacing_spin)
          lines_tab_source_lines_spacing.setLayout(lines_tab_source_lines_spacing.layout)

          lines_tab_source_lines_shifts = QWidget()
          lines_tab_source_lines_shifts.layout = QHBoxLayout()
          lines_tab_source_lines_shifts_label = QLabel(" Shifts ")
          lines_tab_source_lines_shifts_label2 = QLabel(" Inline ")
          lines_tab_source_lines_shifts_label3 = QLabel(" Crossline ")
          self.lines_tab_source_lines_shifts_spin1 = QDoubleSpinBox()
          self.lines_tab_source_lines_shifts_spin2 = QDoubleSpinBox()

          self.lines_tab_source_lines_make_shp_button = QPushButton(" Make shp file")
          
          lines_tab_source_lines_shifts.layout.addWidget(lines_tab_source_lines_shifts_label)
          lines_tab_source_lines_shifts.layout.addStretch()
          lines_tab_source_lines_shifts.layout.addWidget(lines_tab_source_lines_shifts_label2)
          lines_tab_source_lines_shifts.layout.addWidget(self.lines_tab_source_lines_shifts_spin1)
          lines_tab_source_lines_shifts.layout.addStretch()
          lines_tab_source_lines_shifts.layout.addWidget(lines_tab_source_lines_shifts_label3)
          lines_tab_source_lines_shifts.layout.addWidget(self.lines_tab_source_lines_shifts_spin2)       
          lines_tab_source_lines_shifts.setLayout(lines_tab_source_lines_shifts.layout)

          #lines_tab_source_lines_use_file = QWidget()
          #lines_tab_source_lines_use_file.layout = QHBoxLayout()
          #lines_tab_source_lines_use_file_label = QLabel(" receiver lines shp file: ")
          self.lines_tab_source_lines_use_file_edit = QLineEdit()
          self.lines_tab_source_lines_use_file_button = QPushButton(" Select shp file ")
          #lines_tab_source_lines_use_file.layout.addWidget(lines_tab_source_lines_use_file_label)
          #lines_tab_source_lines_use_file.layout.addWidget(self.lines_tab_source_lines_use_file_edit)
          #lines_tab_source_lines_use_file.layout.addWidget(self.lines_tab_source_lines_use_file_button)
          #lines_tab_source_lines_use_file.setLayout(lines_tab_source_lines_use_file.layout)

          self.lines_tab_source_lines_use_file_radiobutton = QRadioButton(" Use lines from file")

          self.lines_tab_source_line_start_number = QWidget()
          self.lines_tab_source_line_start_number_layout = QHBoxLayout()
          self.lines_tab_source_line_start_number_label = QLabel(" Lines starting number ")
          self.lines_tab_source_line_start_number_spin = QSpinBox()
          self.lines_tab_source_line_start_number_spin.setMaximum(999999)
          self.lines_tab_source_line_start_number_spin.setMinimum(-999999)
          self.lines_tab_source_line_start_number_spin.setValue(5000)
          self.lines_tab_source_line_start_number_layout.addWidget(self.lines_tab_source_line_start_number_label)
          self.lines_tab_source_line_start_number_layout.addWidget(self.lines_tab_source_line_start_number_spin)
          self.lines_tab_source_line_start_number.setLayout(self.lines_tab_source_line_start_number_layout)

          self.lines_tab_source_line_inc_number = QWidget()
          self.lines_tab_source_line_inc_number_layout = QHBoxLayout()
          self.lines_tab_source_line_inc_number_label = QLabel(" Lines increment number ")
          self.lines_tab_source_line_inc_number_spin = QSpinBox()
          self.lines_tab_source_line_inc_number_spin.setMaximum(999999)
          self.lines_tab_source_line_inc_number_spin.setMinimum(-999999)
          self.lines_tab_source_line_inc_number_spin.setValue(1)
          self.lines_tab_source_line_inc_number_layout.addWidget(self.lines_tab_source_line_inc_number_label)
          self.lines_tab_source_line_inc_number_layout.addWidget(self.lines_tab_source_line_inc_number_spin)
          self.lines_tab_source_line_inc_number.setLayout(self.lines_tab_source_line_inc_number_layout)
          
          vbox_source_lines.addWidget(radiobutton_new_source_lines)
          vbox_source_lines.addWidget(lines_tab_source_lines_spacing)
          vbox_source_lines.addWidget(lines_tab_source_lines_shifts)
          vbox_source_lines.addWidget(self.lines_tab_source_line_start_number)
          vbox_source_lines.addWidget(self.lines_tab_source_line_inc_number)
          vbox_source_lines.addWidget(self.lines_tab_source_lines_make_shp_button)
          vbox_source_lines.addWidget(self.lines_tab_source_lines_use_file_radiobutton)
          vbox_source_lines.addWidget(self.lines_tab_source_lines_use_file_button)
          vbox_source_lines.addWidget(self.lines_tab_source_lines_use_file_edit)

          #vbox_source_lines.addWidget(lines_tab_source_lines_use_file)

          # put tab 2 together
          lines_tab_2groupbox.layout.addWidget(groupbox_receiver_lines)
          lines_tab_2groupbox.layout.addWidget(groupbox_source_lines)
          lines_tab_2groupbox.setLayout(lines_tab_2groupbox.layout)

          self.lines_tab.layout = QVBoxLayout(self)
          self.lines_tab.layout.addWidget(lines_tab_2groupbox)


          #self.lines_tab_parameters = QWidget()
          #self.lines_tab_parameters.layout = QHBoxLayout()
          #self.lines_tab_parameter_save = QPushButton(" Save parameters ")
          #self.lines_tab_parameter_load = QPushButton(" Load parameters ")
          #self.lines_tab_parameters.layout.addWidget(self.lines_tab_parameter_save)
          #self.lines_tab_parameters.layout.addWidget(self.lines_tab_parameter_load)
          #self.lines_tab_parameters.setLayout(self.lines_tab_parameters.layout)
          #self.lines_tab.layout.addWidget(self.lines_tab_parameters)

          #self.lines_tab.layout.addStretch()

          self.lines_tab.setLayout(self.lines_tab.layout)

          # action tab lines
          #self.lines_tab_parameter_save.clicked.connect(self.save_line_parameters)
          #self.lines_tab_parameter_load.clicked.connect(self.load_line_parameters)

          #self.lines_tab_source_lines_use_file_button.clicked.connect(self.select_source_lines_file)
          #self.lines_tab_receiver_lines_use_file_button.clicked.connect(self.select_receiver_lines_file)

          self.lines_tab_source_lines_make_shp_button.clicked.connect(self.make_source_lines)
          self.lines_tab_receiver_lines_make_shp_button.clicked.connect(self.make_receiver_lines)
          
          #########################
          # tab points - objects
          points_tab_2groupbox = QWidget()
          points_tab_2groupbox.layout = QHBoxLayout()

          # create the two groupbox for receiver and sources
          groupbox_receiver_points = QGroupBox(" Receiver points ")
          groupbox_source_points = QGroupBox(" Source points ")


          # populate the receiver groupbox
          vbox_receiver_points = QVBoxLayout()
          groupbox_receiver_points.setLayout(vbox_receiver_points)
          self.radiobutton_create_receiver_points = QRadioButton(" Create new receiver points ")
          self.radiobutton_create_receiver_points.setChecked(True)
          self.receiver_point_use_file_radiobutton = QRadioButton(" Use receiver points from file ")

          receiver_point_spacing = QWidget()
          receiver_point_spacing_layout = QHBoxLayout()
          receiver_point_spacing_label = QLabel(" Receiver point spacing: ")
          self.receiver_point_spacing_spin = QDoubleSpinBox()
          self.receiver_point_spacing_spin.setMinimum(0.1)
          self.receiver_point_spacing_spin.setMaximum(10000)
          self.receiver_point_spacing_spin.setValue(10)
          receiver_point_spacing_layout.addWidget(receiver_point_spacing_label)
          receiver_point_spacing_layout.addWidget(self.receiver_point_spacing_spin)
          receiver_point_spacing.setLayout(receiver_point_spacing_layout)

          receiver_point_number = QWidget()
          receiver_point_number_layout = QHBoxLayout()
          receiver_point_number_start_label = QLabel(" Receiver point starting number: ")
          self.receiver_point_number_start_spin = QSpinBox()
          self.receiver_point_number_start_spin.setMinimum(0)
          self.receiver_point_number_start_spin.setMaximum(99999999)
          self.receiver_point_number_start_spin.setValue(5000)
          receiver_point_number_layout.addWidget(receiver_point_number_start_label)
          receiver_point_number_layout.addWidget(self.receiver_point_number_start_spin)
          receiver_point_number.setLayout(receiver_point_number_layout)

          receiver_point_inc = QWidget()
          receiver_point_inc_layout = QHBoxLayout()
          receiver_point_inc_label = QLabel(" Receiver point number increment: ")
          self.receiver_point_inc_spin = QSpinBox()
          self.receiver_point_inc_spin.setValue(1)
          self.receiver_point_inc_spin.setMinimum(1)
          self.receiver_point_inc_spin.setMaximum(999999)
          receiver_point_inc_layout.addWidget(receiver_point_inc_label)
          receiver_point_inc_layout.addWidget(self.receiver_point_inc_spin)
          receiver_point_inc.setLayout(receiver_point_inc_layout)

          self.receiver_point_clip_outline_checkbox = QCheckBox(" Clip outline ")
          self.receiver_point_clip_outline_select_file_button = QPushButton(" Select shp file ")
          self.receiver_point_clip_outline_file_edit = QLineEdit()
          self.receiver_point_mask_checkbox = QCheckBox(" Mask exclusion zones ")
          self.receiver_point_mask_select_file_button = QPushButton(" Select shp file ")
          self.receiver_point_mask_file_edit = QLineEdit()
          self.receiver_point_new_file_button = QPushButton(" Make shp file ")
          self.receiver_point_use_file_button = QPushButton(" Select shp file ")
          self.receiver_point_use_file_edit = QLineEdit()

          vbox_receiver_points.addWidget(self.radiobutton_create_receiver_points)
          vbox_receiver_points.addWidget(receiver_point_spacing)
          vbox_receiver_points.addWidget(receiver_point_number)
          vbox_receiver_points.addWidget(receiver_point_inc)
          vbox_receiver_points.addWidget(self.receiver_point_clip_outline_checkbox)
          vbox_receiver_points.addWidget(self.receiver_point_clip_outline_select_file_button)
          vbox_receiver_points.addWidget(self.receiver_point_clip_outline_file_edit)
          vbox_receiver_points.addWidget(self.receiver_point_mask_checkbox)
          vbox_receiver_points.addWidget(self.receiver_point_mask_select_file_button)
          vbox_receiver_points.addWidget(self.receiver_point_mask_file_edit)
          vbox_receiver_points.addWidget(self.receiver_point_new_file_button)

          vbox_receiver_points.addWidget(self.receiver_point_use_file_radiobutton)
          vbox_receiver_points.addWidget(self.receiver_point_use_file_button)
          vbox_receiver_points.addWidget(self.receiver_point_use_file_edit)

          # calls for the receiver points
          self.receiver_point_new_file_button.clicked.connect(self.make_receiver_point)

          # populate teh source groupbox
          vbox_source_points = QVBoxLayout()
          groupbox_source_points.setLayout(vbox_source_points)
          self.source_point_create_radiobutton = QRadioButton(" Create new source points ")
          self.source_point_create_radiobutton.setChecked(True)
          self.source_point_use_file_radiobutton = QRadioButton(" Use source points from file ")

          source_point_spacing = QWidget()
          source_point_spacing_layout = QHBoxLayout()
          source_point_spacing_label = QLabel(" Source point spacing: ")
          self.source_point_spacing_spin = QDoubleSpinBox()
          self.source_point_spacing_spin.setMinimum(-9999999)
          self.source_point_spacing_spin.setMaximum(9999999)
          self.source_point_spacing_spin.setValue(10)

          source_point_spacing_layout.addWidget(source_point_spacing_label)
          source_point_spacing_layout.addWidget(self.source_point_spacing_spin)
          source_point_spacing.setLayout(source_point_spacing_layout)

          source_point_start_number = QWidget()
          source_point_start_number_layout = QHBoxLayout()
          source_point_start_number_label = QLabel(" Source point start number: ")
          self.source_point_start_number_spin = QSpinBox()
          self.source_point_start_number_spin.setMinimum(1)
          self.source_point_start_number_spin.setMaximum(999999)
          self.source_point_start_number_spin.setValue(1000)
          source_point_start_number_layout.addWidget(source_point_start_number_label)
          source_point_start_number_layout.addWidget(self.source_point_start_number_spin)
          source_point_start_number.setLayout(source_point_start_number_layout)

          source_point_inc = QWidget()
          source_point_inc_layout = QHBoxLayout()
          source_point_inc_label = QLabel(" Source point number increment: ")
          self.source_point_inc_spin = QSpinBox()
          self.source_point_inc_spin.setValue(1)
          source_point_inc_layout.addWidget(source_point_inc_label)
          source_point_inc_layout.addWidget(self.source_point_inc_spin)
          source_point_inc.setLayout(source_point_inc_layout)

          self.source_point_mask_checkbox = QCheckBox(" Mask exclusion zones ")
          self.source_point_mask_select_file_button = QPushButton(" Select shp file ")
          self.source_point_mask_file_edit = QLineEdit()
          self.source_point_new_file_button = QPushButton(" Make shp file ")
          self.source_point_clip_outline_checkbox = QCheckBox(" Clip outline ")
          self.source_point_clip_outline_select_file_button = QPushButton(" Select shp file ")
          self.source_point_clip_outline_file_edit = QLineEdit()

          self.source_point_use_file_button = QPushButton(" Select shp file ")
          self.source_point_use_file_edit = QLineEdit()

          vbox_source_points.addWidget(self.source_point_create_radiobutton)
          vbox_source_points.addWidget(source_point_spacing)
          vbox_source_points.addWidget(source_point_start_number)
          vbox_source_points.addWidget(source_point_inc)
          vbox_source_points.addWidget(self.source_point_clip_outline_checkbox)
          vbox_source_points.addWidget(self.source_point_clip_outline_select_file_button)
          vbox_source_points.addWidget(self.source_point_clip_outline_file_edit)
          vbox_source_points.addWidget(self.source_point_mask_checkbox)
          vbox_source_points.addWidget(self.source_point_mask_select_file_button)
          vbox_source_points.addWidget(self.source_point_mask_file_edit)
          vbox_source_points.addWidget(self.source_point_new_file_button)
          vbox_source_points.addWidget(self.source_point_use_file_radiobutton)
          vbox_source_points.addWidget(self.source_point_use_file_button)
          vbox_source_points.addWidget(self.source_point_use_file_edit)


          # calls for the source points
          self.source_point_new_file_button.clicked.connect(self.make_source_point)
          points_tab_2groupbox.layout.addWidget(groupbox_receiver_points)
          points_tab_2groupbox.layout.addWidget(groupbox_source_points)
          points_tab_2groupbox.setLayout(points_tab_2groupbox.layout)

          self.points_tab.layout = QVBoxLayout(self)
          self.points_tab.layout.addWidget(points_tab_2groupbox)
          self.points_tab.layout.addStretch()
          self.points_tab.setLayout(self.points_tab.layout)
          
          # TAB 3 - GRID AND FOLD MAP
          # 
          foldmap_widget = QWidget()
          foldmap_widget_layout = QHBoxLayout()

          # create the two groupbox for grid and foldmap
          grid_groupbox = QGroupBox(" Grid ")

          self.grid_make_new_radiobutton = QRadioButton(" Make new grid ")
          self.grid_make_new_radiobutton.setChecked(True)

          grid_inline_bin = QWidget()
          grid_inline_bin_layout = QHBoxLayout()

          grid_inline_label = QLabel("Inline bin size: ")
          self.grid_inline_bin_spin = QSpinBox()
          self.grid_inline_bin_spin.setMinimum(1)
          self.grid_inline_bin_spin.setMaximum(999999)
          self.grid_inline_bin_spin.setValue(10)

          grid_inline_bin_layout.addWidget(grid_inline_label)
          grid_inline_bin_layout.addWidget(self.grid_inline_bin_spin)
          grid_inline_bin.setLayout(grid_inline_bin_layout)

          grid_crossline_bin = QWidget()
          grid_crossline_bin_layout = QHBoxLayout()

          grid_crossline_label = QLabel("Crossline bin size: ")
          self.grid_crossline_bin_spin = QSpinBox()
          self.grid_crossline_bin_spin.setMinimum(1)
          self.grid_crossline_bin_spin.setMaximum(999999)
          self.grid_crossline_bin_spin.setValue(10)

          grid_crossline_bin_layout.addWidget(grid_crossline_label)
          grid_crossline_bin_layout.addWidget(self.grid_crossline_bin_spin)
          grid_crossline_bin.setLayout(grid_crossline_bin_layout)

          self.grid_make_new_button = QPushButton(" Make shp file ")
          self.grid_make_new_button.clicked.connect(self.make_grid)

          self.grid_use_file_radiobutton = QRadioButton(" Use grid file ")
          self.grid_use_file_button = QPushButton(" Select shp file ")
          self.grid_use_file_edit = QLineEdit()

          grid_groupbox_layout = QVBoxLayout()
          grid_groupbox_layout.addWidget(self.grid_make_new_radiobutton)
          grid_groupbox_layout.addWidget(grid_inline_bin)
          grid_groupbox_layout.addWidget(grid_crossline_bin)
          grid_groupbox_layout.addWidget(self.grid_make_new_button)
          grid_groupbox_layout.addWidget(self.grid_use_file_radiobutton)
          grid_groupbox_layout.addWidget(self.grid_use_file_button)
          grid_groupbox_layout.addWidget(self.grid_use_file_edit)

          grid_groupbox.setLayout(grid_groupbox_layout)






          fold_groupbox = QGroupBox(" Fold ")
          fold_groupbox_layout = QVBoxLayout()
          fold_offset = QWidget()
          fold_offset_layout = QHBoxLayout()
          fold_offset_label = QLabel("Maximum offset: ")
          self.fold_offset_spin = QSpinBox()
          self.fold_offset_spin.setMinimum(1)
          self.fold_offset_spin.setMaximum(9999999)
          self.fold_offset_spin.setValue(500)
          fold_offset_layout.addWidget(fold_offset_label)
          fold_offset_layout.addWidget(self.fold_offset_spin)
          fold_offset.setLayout(fold_offset_layout)

          self.fold_cmp_radiobutton = QRadioButton(" Calculate cmp ")
          self.fold_cmp_radiobutton.setChecked(True)
          self.fold_cmp_checkbox = QCheckBox(" Save cmp as shp (large) ")
          self.fold_azimuth_checkbox = QCheckBox(" Calculate azimuth (slow) ")

          partitioning_widget = QWidget()
          partitioning_widget_layout = QHBoxLayout()
          partitioning_label = QLabel(" Partitioning ")
          self.partitioning_spin = QSpinBox()
          self.partitioning_spin.setMinimum(1)
          self.partitioning_spin.setMaximum(2147483647)
          self.partitioning_spin.setValue(100)
          partitioning_widget_layout.addWidget(partitioning_label)
          partitioning_widget_layout.addWidget(self.partitioning_spin)
          partitioning_widget.setLayout(partitioning_widget_layout)



          self.fold_button = QPushButton(" Calculate fold ")
          self.fold_button.clicked.connect(self.calculate_fold)
          fold_groupbox_layout.addWidget(self.fold_cmp_radiobutton)
          fold_groupbox_layout.addWidget(fold_offset)
          fold_groupbox_layout.addWidget(partitioning_widget)

          fold_groupbox_layout.addWidget(self.fold_cmp_checkbox)
          fold_groupbox_layout.addWidget(self.fold_azimuth_checkbox)

          self.fold_cmp_file_radiobutton = QRadioButton(" Use shp file for cmp ")
          self.fold_cmp_file_button = QPushButton(" Select shp file ")
          self.fold_cmp_file_edit = QLineEdit()
          fold_groupbox_layout.addWidget(self.fold_cmp_file_radiobutton)
          fold_groupbox_layout.addWidget(self.fold_cmp_file_button)
          fold_groupbox_layout.addWidget(self.fold_cmp_file_edit)

          fold_groupbox_layout.addStretch()
          fold_groupbox_layout.addWidget(self.fold_button)
          fold_groupbox.setLayout(fold_groupbox_layout)

          foldmap_widget_layout.addWidget(grid_groupbox)
          foldmap_widget_layout.addWidget(fold_groupbox)
          foldmap_widget.setLayout(foldmap_widget_layout)

          self.foldmap_tab_layout = QVBoxLayout(self)
          self.foldmap_tab_layout.addWidget(foldmap_widget)
          self.foldmap_tab_layout.addStretch()
          self.foldmap_tab.setLayout(self.foldmap_tab_layout)

          
          
          
          
          
          
          
          
          # outside the tabs
          # progress bar
          self.pbar = QProgressBar(self)
          self.pbar.setValue(0)

          self.statusbar = QStatusBar(self)
          self.statusbar.showMessage('Ready', 10000)

          self.main_vlayout = QVBoxLayout()
          self.main_vlayout.addWidget(self.tabs)
          self.main_vlayout.addWidget(parameters)
          self.main_vlayout.addWidget(self.pbar)
          self.main_vlayout.addWidget(self.statusbar)
          self.setLayout(self.main_vlayout)

          # functions

     def cmp_assembled(self):
          self.pbar.setValue(0) 
          self.statusbar.showMessage('Task completed', 5000)  

     def cmp_completed(self):
          global proj_dict, points
          self.pbar.setValue(0)
          self.statusbar.showMessage('Task completed', 5000)

     def calculate_fold(self):
          global proj_dict

          self.save_project_parameters()
          self.statusbar.showMessage('Calculating cmp positions',5000)

          self.cmp_worker = CMPWorker()  
          self.cmp_worker.countChanged.connect(self.progress)
          self.cmp_worker.partChanged.connect(self.cmp_part_indicator)
          self.cmp_worker.finished.connect(self.cmp_completed)
          self.cmp_worker.start() 

               

     def cmp_part_indicator(self,value):
          if value ==1:
               self.statusbar.showMessage('Calculating the positions of the cmps', 100000)
          elif value==2:
               self.statusbar.showMessage('Assembling the cmps', 100000)
          elif value == 3:
               self.fold_cmp_file_radiobutton.setChecked(True)
               self.fold_cmp_file_edit.setText(os.path.join(proj_dict["dir"], proj_dict["option"],'cmp.zip'))
               self.statusbar.showMessage('Calculating the fold', 100000)


     def make_receiver_point(self):
          global points
          points["lines_file"] = proj_dict["receiver_line_file"]
          points["spacing"] = self.receiver_point_spacing_spin.value()
          points["pointtype"] = 'receiver'

          with open('points.json', 'w') as fp:
                json.dump(points, fp)

          self.point_worker = PointWorker()  
          self.point_worker.countChanged.connect(self.progress)
          self.point_worker.finished.connect(self.make_points_completed)
          self.point_worker.start() 

     def make_source_point(self):
          global points
          points["lines_file"] = proj_dict["source_line_file"]
          points["spacing"] = self.source_point_spacing_spin.value()
          points["pointtype"] = 'source'

          with open('points.json', 'w') as fp:
                json.dump(points, fp)

          self.point_worker = PointWorker()  
          self.point_worker.countChanged.connect(self.progress)
          self.point_worker.finished.connect(self.make_points_completed)
          self.point_worker.start() 

     def make_points_completed(self): 
          global proj_dict, points
          self.pbar.setValue(0)
          self.statusbar.showMessage('Task completed', 5000)

          pointtype = points["pointtype"]

          if pointtype == 'source':
               self.source_point_use_file_edit.setText(proj_dict["source_point_file"]) 
               self.source_point_use_file_radiobutton.setChecked(True)

               proj_dict["source_point_spacing"] = self.source_point_spacing_spin.value()
               proj_dict["source_point_start_number"] = self.source_point_start_number_spin.value()
               proj_dict["source_point_inc_number"] = self.source_point_inc_spin.value()

          else:
               self.receiver_point_use_file_edit.setText(proj_dict["receiver_point_file"]) 
               self.receiver_point_use_file_radiobutton.setChecked(True)

               proj_dict["receiver_point_spacing"] = self.receiver_point_spacing_spin.value()
               proj_dict["receiver_point_start_number"] = self.receiver_point_number_start_spin.value()
               proj_dict["receiver_point_inc_number"] = self.receiver_point_inc_spin.value()


          self.save_project_parameters()  

     def make_source_lines(self):
          global lines, proj_dict

          lines["type"] = 'source'
          lines["length"] = proj_dict["crossline_length"]
          lines["width"] = proj_dict["inline_length"]
          lines["azimuth"] = float(proj_dict["inline_azimuth"]) +90
          lines["spacing"] = self.lines_tab_source_lines_spacing_spin.value()
          lines["inline_shift"] = self.lines_tab_source_lines_shifts_spin1.value()
          lines["crossline_shift"] = self.lines_tab_source_lines_shifts_spin2.value()
          lines["start_number"] = self.lines_tab_source_line_start_number_spin.value()
          lines["inc_number"] = self.lines_tab_source_line_inc_number_spin.value()

          with open('lines.json', 'w') as fp:
                json.dump(lines, fp)

          self.line_worker = LinesWorker()  
          self.line_worker.countChanged.connect(self.progress)
          self.line_worker.finished.connect(self.make_lines_completed)
          self.line_worker.start()
          
     def make_receiver_lines(self):
          global lines, proj_dict

          lines["type"] = 'receiver'
          lines["length"] = proj_dict["inline_length"]
          lines["width"] = proj_dict["crossline_length"]
          lines["azimuth"] = float(proj_dict["inline_azimuth"]) 
          lines["spacing"] = self.lines_tab_receiver_lines_spacing_spin.value()
          lines["inline_shift"] = self.lines_tab_receiver_lines_shifts_spin1.value()
          lines["crossline_shift"] = self.lines_tab_receiver_lines_shifts_spin2.value()
          lines["start_number"] = self.lines_tab_receiver_line_start_number_spin.value()
          lines["inc_number"] = self.lines_tab_receiver_line_inc_number_spin.value()

          with open('lines.json', 'w') as fp:
                json.dump(lines, fp)

          self.line_worker = LinesWorker()  
          self.line_worker.countChanged.connect(self.progress)
          self.line_worker.finished.connect(self.make_lines_completed)
          self.line_worker.start()          

     def pick_project_folder(self):
        global proj_dict
        dir = QFileDialog.getExistingDirectory()  
        self.project_folder_edit.setText(dir)
        proj_dict["dir"] = dir
        self.statusbar.showMessage('Project folder set', 5000)     

     def save_project_parameters(self):
          global proj_dict

          # tab 1
          dir = self.project_folder_edit.text()
          job = self.job_number_edit.text()
          client = self.client_name_edit.text()
          project = self.project_name_edit.text()
          option = self.option_name_edit.text()
          dict_name = job+'_'+client+'_'+project+'_'+option+'_dictionary.json'

          proj_dict["dict_name"] = dict_name
          proj_dict["dir"] = dir
          proj_dict["job"] = job 
          proj_dict["client"] = client  
          proj_dict["project"] = project 
          proj_dict["option"] = option 
          proj_dict["epsg"] = self.epsg_name_edit.text()

          proj_dict["origin"] = self.survey_origin_edit.text()
          proj_dict["inline_length"] = self.survey_area_inline_length_spin.value()
          proj_dict["inline_azimuth"] = self.survey_area_inline_azimuth_spin.value()
          proj_dict["crossline_length"] = self.survey_area_crossline_length_spin.value()

          # tab 2 - lines
          proj_dict["receiver_line_file_exist"] = self.lines_tab_receiver_use_file_radiobutton.isChecked()
          proj_dict["receiver_line_file"] = self.lines_tab_receiver_lines_use_file_edit.text()
          proj_dict["receiver_line_spacing"] = self.lines_tab_receiver_lines_spacing_spin.value()
          proj_dict["receiver_line_inline_shift"] = self.lines_tab_receiver_lines_shifts_spin1.value() 
          proj_dict["receiver_line_crossline_shift"] = self.lines_tab_receiver_lines_shifts_spin2.value()
          proj_dict["receiver_line_start_number"] = self.lines_tab_receiver_line_start_number_spin.value()
          proj_dict["receiver_line_inc_number"] = self.lines_tab_receiver_line_inc_number_spin.value()

          proj_dict["source_line_file_exist"] = self.lines_tab_source_lines_use_file_radiobutton.isChecked()
          proj_dict["source_line_file"] = self.lines_tab_source_lines_use_file_edit.text()
          proj_dict["source_line_spacing"] = self.lines_tab_source_lines_spacing_spin.value()
          proj_dict["source_line_inline_shift"] = self.lines_tab_source_lines_shifts_spin1.value() 
          proj_dict["source_line_crossline_shift"] = self.lines_tab_source_lines_shifts_spin2.value()
          proj_dict["source_line_start_number"] = self.lines_tab_source_line_start_number_spin.value()
          proj_dict["source_line_inc_number"] = self.lines_tab_source_line_inc_number_spin.value()

          # tab 3 - points
          proj_dict["receiver_point_file_exist"] = self.receiver_point_use_file_radiobutton.isChecked()
          proj_dict["receiver_point_file"] = self.receiver_point_use_file_edit.text()
          proj_dict["receiver_point_spacing"] = self.receiver_point_spacing_spin.value()
          proj_dict["receiver_point_start_number"] = self.receiver_point_number_start_spin.value()
          proj_dict["receiver_point_inc_number"] = self.receiver_point_inc_spin.value()
          proj_dict["receiver_point_clip_outline"] = self.receiver_point_clip_outline_checkbox.isChecked()
          proj_dict["receiver_point_clip_file"] = self.receiver_point_clip_outline_file_edit.text()
          proj_dict["receiver_point_mask"] = self.receiver_point_mask_checkbox.isChecked()
          proj_dict["receiver_point_mask_file"] = self.receiver_point_mask_file_edit.text()

          
          proj_dict["source_point_file_exist"] = self.source_point_use_file_radiobutton.isChecked()
          proj_dict["source_point_file"] = self.source_point_use_file_edit.text()
          proj_dict["source_point_spacing"] = self.source_point_spacing_spin.value()
          proj_dict["source_point_start_number"] = self.source_point_start_number_spin.value()
          proj_dict["source_point_inc_number"] = self.source_point_inc_spin.value()
          proj_dict["source_point_clip_outline"] = self.source_point_clip_outline_checkbox.isChecked()
          proj_dict["source_point_clip_file"] = self.source_point_clip_outline_file_edit.text()
          proj_dict["source_point_mask"] = self.source_point_mask_checkbox.isChecked()
          proj_dict["source_point_mask_file"] = self.source_point_mask_file_edit.text()

          # tab 4 - grid & fold
          proj_dict["grid_file_exist"] = self.grid_use_file_radiobutton.isChecked()
          proj_dict["grid_file"] = self.grid_use_file_edit.text()
          proj_dict["inline_bin_size"] = self.grid_inline_bin_spin.value()
          proj_dict["crossline_bin_size"] = self.grid_crossline_bin_spin.value()

          proj_dict["cmp_file_exist"] = self.fold_cmp_file_radiobutton.isChecked()
          proj_dict["cmp_file"] = self.fold_cmp_file_edit.text()
          proj_dict["fold_max_offset"] = self.fold_offset_spin.value()
          proj_dict["save_cmp"] = self.fold_cmp_checkbox.isChecked()
          proj_dict["calculate_azimuth"] = self.fold_azimuth_checkbox.isChecked()
          proj_dict["partitioning"] = self.partitioning_spin.value()

          with open(os.path.join(dir,dict_name), 'w') as fp:
                         json.dump(proj_dict, fp)

          self.statusbar.showMessage("Saving parameters",1500)

          if os.path.exists(os.path.join(dir,option)):
               pass
          else:
               os.mkdir(os.path.join(dir,option))
  

     def load_project_parameters(self):
          global proj_dict

          file = QFileDialog.getOpenFileName()[0]

          if Path(file).suffix == '.json':
               self.statusbar.showMessage('Loading '+file, 5000)
               with open(file) as json_file:
                    content = json_file.read()
                    parsed = json.loads(content)

                    if 'dir' in content:
                         directory = parsed["dir"]
                         self.project_folder_edit.setText(directory) 

                    if 'job' in content:
                         job = parsed["job"]
                         self.job_number_edit.setText(job)

                    if 'client' in content:
                         client = parsed["client"]
                         self.client_name_edit.setText(client)  

                    if 'project' in content:
                         project = parsed["project"]
                         self.project_name_edit.setText(project)  

                    if 'option' in content:
                         option = parsed["option"]
                         self.option_name_edit.setText(option)  

                    if 'epsg' in content:
                         epsg = parsed["epsg"]
                         self.epsg_name_edit.setText(str(epsg))    

                    if 'dict_name' in content:
                         dict_name = parsed["dict_name"]

                    if 'origin' in content:
                         origin = parsed["origin"]
                         self.survey_origin_edit.setText(origin)

                    if 'inline_length' in content:
                         inline_length = parsed["inline_length"]
                         self.survey_area_inline_length_spin.setValue(int(inline_length))

                    if 'inline_azimuth' in content:
                         inline_azimuth = parsed["inline_azimuth"]
                         self.survey_area_inline_azimuth_spin.setValue(float(inline_azimuth))

                    if 'crossline_length' in content:
                         crossline_length = parsed["crossline_length"]
                         self.survey_area_crossline_length_spin.setValue(int(crossline_length))

                    if 'receiver_line_file_exist' in content:
                         self.lines_tab_receiver_use_file_radiobutton.setChecked(True)

                    if 'receiver_line_file' in content:
                         self.lines_tab_receiver_lines_use_file_edit.setText(parsed["receiver_line_file"])
                    
                    if 'receiver_line_spacing' in content:
                         self.lines_tab_receiver_lines_spacing_spin.setValue(parsed["receiver_line_spacing"])
                    
                    if 'receiver_line_inline_shift' in content:
                         self.lines_tab_receiver_lines_shifts_spin1.setValue(parsed["receiver_line_inline_shift"])

                    if 'receiver_line_crossline_shift' in content:
                         self.lines_tab_receiver_lines_shifts_spin2.setValue(parsed["receiver_line_crossline_shift"])

                    if 'receiver_line_start_number' in content:
                         self.lines_tab_receiver_line_start_number_spin.setValue(parsed["receiver_line_start_number"])  

                    if 'receiver_line_inc_number' in content:
                         self.lines_tab_receiver_line_inc_number_spin.setValue(parsed["receiver_line_inc_number"])   

                    if 'source_line_file_exist' in content:
                         self.lines_tab_source_lines_use_file_radiobutton.setChecked(True)

                    if 'source_line_file' in content:
                         self.lines_tab_source_lines_use_file_edit.setText(parsed["source_line_file"])
                    
                    if 'source_line_spacing' in content:
                         self.lines_tab_source_lines_spacing_spin.setValue(parsed["source_line_spacing"])
                    
                    if 'source_line_inline_shift' in content:
                         self.lines_tab_source_lines_shifts_spin1.setValue(parsed["source_line_inline_shift"])

                    if 'source_line_crossline_shift' in content:
                         self.lines_tab_source_lines_shifts_spin2.setValue(parsed["source_line_crossline_shift"])

                    if 'source_line_start_number' in content:
                         self.lines_tab_source_line_start_number_spin.setValue(parsed["source_line_start_number"])  

                    if 'source_line_inc_number' in content:
                         self.lines_tab_source_line_inc_number_spin.setValue(parsed["source_line_inc_number"])  

                    # load the point parameters
                    if 'receiver_point_file_exist' in content:
                         self.receiver_point_use_file_radiobutton.setChecked(True) 

                    if 'receiver_point_file' in content:
                         self.receiver_point_use_file_edit.setText(parsed["receiver_point_file"])
                    
                    if 'receiver_point_spacing' in content:
                         self.receiver_point_spacing_spin.setValue(parsed["receiver_point_spacing"])

                    if 'receiver_point_start_number' in content:
                         self.receiver_point_number_start_spin.setValue(parsed["receiver_point_start_number"])

                    if 'receiver_point_inc_number' in content:
                         self.receiver_point_inc_spin.setValue(parsed["receiver_point_inc_number"])

                    if 'source_point_file_exist' in content:
                         self.source_point_use_file_radiobutton.setChecked(True) 

                    if 'source_point_file' in content:
                         self.source_point_use_file_edit.setText(parsed["source_point_file"])
                    
                    if 'source_point_spacing' in content:
                         self.source_point_spacing_spin.setValue(parsed["source_point_spacing"])

                    if 'source_point_start_number' in content:
                         self.source_point_start_number_spin.setValue(parsed["source_point_start_number"])

                    if 'source_point_inc_number' in content:
                         self.source_point_inc_spin.setValue(parsed["source_point_inc_number"])
                    if 'grid_file_exist' in content:
                         if parsed["grid_file_exist"]:
                              self.grid_use_file_radiobutton.setChecked(True)

                    if 'grid_file' in content:
                         self.grid_use_file_edit.setText(parsed["grid_file"])

                    if 'inline_bin_size' in content:
                         self.grid_inline_bin_spin.setValue(parsed["inline_bin_size"])

                    if 'crossline_bin_size' in content:
                         self.grid_crossline_bin_spin.setValue(parsed["crossline_bin_size"])


                    if 'cmp_file_exist' in content:
                         self.fold_cmp_file_radiobutton.setChecked(parsed["cmp_file_exist"])     

                    if 'cmp_file' in content:
                         self.fold_cmp_file_edit.setText(parsed["cmp_file"])

                    if 'fold_max_offset' in content:
                         self.fold_offset_spin.setValue(parsed["fold_max_offset"])

                    if 'save_cmp' in content:
                         self.fold_cmp_checkbox.setChecked(parsed["save_cmp"])

                    if 'calculate_azimuth' in content:
                         self.fold_azimuth_checkbox.setChecked(parsed["calculate_azimuth"])

                    if 'partitioning' in content:
                         self.partitioning_spin.setValue(parsed["partitioning"])
                  
               self.save_project_parameters()      
                  
          else:
             self.statusbar.showMessage('Selected file is not a dictionary', 5000)

     def make_origin(self):
          global proj_dict
          proj_folder = proj_dict["dir"]
          option = proj_dict["option"]

          origin = proj_dict["origin"].split(",")
          origin_x = float(origin[0])
          origin_y = float(origin[1])

          epsg = int(proj_dict["epsg"])

          point  = []
          point.append(Point([origin_x, origin_y]))
          origin_point = gpd.GeoDataFrame(geometry=point, crs='epsg:'+str(epsg))

          if os.path.exists(os.path.join(proj_folder, option, 'origin')):
               pass
          else:
               os.makedirs(os.path.join(proj_folder, option, 'origin'))

          origin_point.to_file(os.path.join(proj_folder, option, 'origin', 'origin.shp'))
          self.make_archive(os.path.join(proj_folder, option, 'origin'), os.path.join(proj_folder, option, 'origin.zip'))
          shutil.rmtree(os.path.join(proj_folder, option, 'origin'))

          self.statusbar.showMessage('Origin saved as shapefile', 5000)

     def make_archive(self, source, destination):
          base_name = '.'.join(destination.split('.')[:-1])
          format = destination.split('.')[-1]
          root_dir = os.path.dirname(source)
          base_dir = os.path.basename(source.strip(os.sep))
          shutil.make_archive(base_name, format, root_dir, base_dir)

     def make_outline(self):
          global proj_dict

          # save the project parameters to insure we have captured all changes
          self.save_project_parameters()

          proj_folder = proj_dict["dir"]
          option = proj_dict["option"]
          inline_length = proj_dict["inline_length"]
          inline_azimuth = proj_dict["inline_azimuth"]
          crossline_length = proj_dict["crossline_length"]
          epsg = int(proj_dict["epsg"])
          origin = proj_dict["origin"].split(",")
          origin_x = float(origin[0])
          origin_y = float(origin[1])

          point_1 = Point([origin_x, origin_y])
          point_2 = Point([origin_x + inline_length * np.cos((90-inline_azimuth)*np.pi/180), origin_y+ inline_length * np.sin((90-inline_azimuth)*np.pi/180)])
          point_3 = Point([point_2.x + crossline_length * np.cos(-inline_azimuth*np.pi/180), point_2.y + crossline_length * np.sin(-inline_azimuth*np.pi/180)])
          point_4 = Point([[point_3.x - inline_length * np.cos((90-inline_azimuth)*np.pi/180), point_3.y - inline_length * np.sin((90-inline_azimuth)*np.pi/180)]])

          points = [point_1, point_2, point_3, point_4, point_1]
          poly = Polygon([[p.x, p.y] for p in points])
          outline = gpd.GeoDataFrame(geometry=[poly], crs='epsg:'+str(epsg))

          if os.path.exists(os.path.join(proj_folder, option, 'outline')):
               pass
          else:
               os.makedirs(os.path.join(proj_folder, option, 'outline'))

          outline.to_file(os.path.join(proj_folder, option, 'outline', 'outline.shp'))
          self.make_archive(os.path.join(proj_folder, option, 'outline'), os.path.join(proj_folder, option, 'outline.zip'))
          shutil.rmtree(os.path.join(proj_folder, option, 'outline'))

          self.statusbar.showMessage('Outline saved as shapefile', 5000)

     def progress(self, value):
        
        Current_message = self.statusbar.currentMessage()
        self.statusbar.showMessage(Current_message, 5000000)
        self.pbar.setValue(value)
        
         
     def make_grid_completed(self):
          global proj_dict
          self.pbar.setValue(0)
          self.statusbar.showMessage('Task completed - Grid created and saved as grd.shp', 5000)
          
          self.grid_use_file_radiobutton.setChecked(True)
          self.grid_use_file_edit.setText(os.path.join(proj_dict["dir"], proj_dict["option"], 'grd.zip'))
          proj_dict["grid_file_exist"] = True
          proj_dict["grid_file"] = os.path.join(proj_dict["dir"], proj_dict["option"], 'grd.zip')
          proj_dict["inline_bin_size"] = self.grid_inline_bin_spin.value()
          proj_dict["crossline_bin_size"] = self.grid_crossline_bin_spin.value()


     def make_lines_completed(self):
          global proj_dict
          self.pbar.setValue(0)
          self.statusbar.showMessage('Task completed', 5000)

          
          if "source_line_file_exist" in proj_dict:
               if proj_dict["source_line_file_exist"]:
                    if self.lines_tab_source_lines_use_file_radiobutton.isChecked():
                         pass
                    else:
                         self.lines_tab_source_lines_use_file_radiobutton.setChecked(True)
                         self.lines_tab_source_lines_use_file_edit.setText(proj_dict["source_line_file"])

                         proj_dict["source_line_spacing"] = self.lines_tab_source_lines_spacing_spin.value()
                         proj_dict["source_line_inline_shift"] = self.lines_tab_source_lines_shifts_spin1.value() 
                         proj_dict["source_line_crossline_shift"] = self.lines_tab_source_lines_shifts_spin2.value()
                         proj_dict["source_line_start_number"] = self.lines_tab_source_line_start_number_spin.value()
                         proj_dict["source_line_inc_number"] = self.lines_tab_source_line_inc_number_spin.value()

                         self.save_project_parameters()
               else:
                    pass

          if "receiver_line_file_exist" in proj_dict:
               if proj_dict["receiver_line_file_exist"]:
                    if self.lines_tab_receiver_use_file_radiobutton.isChecked():
                         pass
                    else:
                         self.lines_tab_receiver_use_file_radiobutton.setChecked(True)
                         self.lines_tab_receiver_lines_use_file_edit.setText(proj_dict["receiver_line_file"])

                         proj_dict["receiver_line_spacing"] = self.lines_tab_receiver_lines_spacing_spin.value()
                         proj_dict["receiver_line_inline_shift"] = self.lines_tab_receiver_lines_shifts_spin1.value() 
                         proj_dict["receiver_line_crossline_shift"] = self.lines_tab_receiver_lines_shifts_spin2.value()
                         proj_dict["receiver_line_start_number"] = self.lines_tab_receiver_line_start_number_spin.value()
                         proj_dict["receiver_line_inc_number"] = self.lines_tab_receiver_line_inc_number_spin.value()

                         self.save_project_parameters()

               else:
                    pass

     def make_grid(self):
          global proj_dict, grid

          proj_dict["inline_bin_size"] = self.grid_inline_bin_spin.value()
          proj_dict["crossline_bin_size"] = self.grid_crossline_bin_spin.value()

          self.grid_worker = GridMaker()  
          self.grid_worker.countChanged.connect(self.progress)
          self.grid_worker.finished.connect(self.make_grid_completed)
          self.grid_worker.start()




if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    window = MainWindow()
    #window.showMaximized()
    window.show()
    exit_code = appctxt.app.exec()      # 2. Invoke appctxt.app.exec()
    sys.exit(exit_code)