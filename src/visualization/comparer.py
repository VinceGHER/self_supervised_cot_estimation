from functools import partial
import os

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from src.experiment import Experiment
from src.robot import Robot
from src.tools import check_file_path, filename_to_timestamp
import open3d as o3d
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import mpl_toolkits.axes_grid1
import matplotlib.widgets
class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, interval=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.2, 0.12), **kwargs):
        self.i = 0
        self.min=mini
        self.max=maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self,self.fig, self.update, interval=interval,frames=self.play(),blit=False,
                                           init_func=init_func, fargs=fargs,cache_frame_data=True,
                                           save_count=save_count, **kwargs )    

    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()
    def backward(self, event=None):
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i+=1
        elif self.i == self.max and not self.forwards:
            self.i-=1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0],pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(sliderax, '', 
                                                self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self,i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self,i):
        # print(self.runs)
        self.slider.set_val(i)
        r = self.func(i)
        # r.extend([self.playerax])
        return r

class Comparer():
    def __init__(self, min_cost_of_transport=0.5, max_cost_of_transport=10,speed_ms=200):
        self.min_cost_of_transport =min_cost_of_transport
        self.max_cost_of_transport =max_cost_of_transport
        self.colorbar1 = None
        self.colorbar2 = None
        self.speed_ms = speed_ms


    def compare(self, exp: Experiment,save_video=False):
        masks= os.listdir(exp.mask_path)
        masks.sort()
        fig, axes = plt.subplots(1, 2,figsize=(12.8,7.2))  # Create subplots once
        ax0, ax1 = axes
    
        plt.title("Comparison of mask and image")
        def plot_image(i):
            name = masks[i]
            timestamp = filename_to_timestamp(name)
            mask = np.load(check_file_path(exp.mask_path, name))
            image= plt.imread(check_file_path(exp.rgb_path, timestamp+".jpg"))
            projected = np.load(check_file_path(exp.projected_path, timestamp+".npy"))
            masked_data_projected = np.ma.masked_where(projected <= 0, projected)
            masked_data_mask = np.ma.masked_where(mask <= 0, mask)
     
            if self.colorbar1 is not None:
                self.colorbar1.remove()
            if self.colorbar2 is not None:
                self.colorbar2.remove()

            ax0.clear()
            ax1.clear()
            im00 = ax0.imshow(image)
            im0 = ax0.imshow(masked_data_projected, alpha=1,cmap='nipy_spectral',vmin=self.min_cost_of_transport,vmax=self.max_cost_of_transport)
            ax0.set_title("Image")
            ax0.axis('off')
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            self.colorbar1 = fig.colorbar(im0, cax=cax, orientation='vertical',label='Cost of transport')

            im11 = ax1.imshow(image)
            im1=ax1.imshow(masked_data_mask, alpha=1,cmap='nipy_spectral',vmin=self.min_cost_of_transport,vmax=self.max_cost_of_transport)
            ax1.set_title("Mask")
            ax1.axis('off')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            self.colorbar2 = fig.colorbar(im1, cax=cax, orientation='vertical',label='Cost of transport')

            return [im0,im1]

        ani = Player(fig, plot_image, maxi=len(masks)-1,interval=self.speed_ms,save_count=len(masks))
        if save_video:
            print("Saving video...")
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"visualisation_{date}.mp4"
            ani.save(filename=os.path.join(exp.get_experiment_folder(), filename), writer="ffmpeg")
            print(f"Video saved as {filename}")
        # ani = animation.FuncAnimation(
        #     fig,  update, interval=500, frames=len(masks), blit=True)


        # axcolor = 'lightgoldenrodyellow'
        # axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        # sfreq = Slider(axfreq, 'Image', 0, len(masks)-1, valinit=0,valstep=1)
        # def update(val):
        #     self.plot_image(fig, exp.mask_path, exp.rgb_path,exp.projected_path,masks[int(val)])
        #     fig.canvas.draw_idle()
        # sfreq.on_changed(update)
        plt.show()

    def plot_robot_camera(self,robot: Robot):
        # Plot robot camera frame
        pose_body = o3d.geometry.TriangleMesh.create_coordinate_frame()
        pose_camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # apply the transformation
        pose_camera.transform(robot.robot_body_to_camera)
        robot_body = o3d.geometry.TriangleMesh.create_box(robot.width,robot.length,robot.height,)
        robot_body.translate(np.array([-robot.width/2,-robot.length/2,-robot.height]))

        o3d.visualization.draw_geometries([pose_body,robot_body,pose_camera])

    def compute_bias_and_variance(self, exp: Experiment):
        masks = os.listdir(exp.mask_path)
        costs = []
        for mask in masks:
            mask_path = check_file_path(exp.mask_path, mask)
            cost = np.load(mask_path)
            # filter where cost is not zero
            cost = cost[cost > 0]
            mean = np.mean(cost)
            if np.isnan(mean):
                continue
            costs.append(np.mean(cost))
        # compute bias and variance
        bias = np.mean(np.array(costs))
        variance = np.var(np.array(costs))
        print(f"Bias: {bias}")
        print(f"Variance: {variance}")

        # normalize the costs
        cost_norm  = (costs - bias)/variance
        # histogram = plt.hist(costs, bins=50)
        plt.hist(cost_norm, bins=50)
        plt.xlabel('Cost of transport')
        plt.ylabel('Frequency')
        plt.title('Histogram of Cost of Transport')
        plt.show()