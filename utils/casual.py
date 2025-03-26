import numpy as np
from seismic_canvas import (SeismicCanvas, volume_slices, XYZAxis, Colorbar)
from vispy.color import Colormap
from vispy import app

def fault3D(seismic_path,fault_path):

    slicing = {'x_pos': 15, 'y_pos': 15, 'z_pos': 407}
    canvas_params = {'size': (900, 900),
                    'axis_scales': (0.5, 0.5, 0.5), # stretch z-axis
                    'colorbar_region_ratio': 0.1,
                    'fov': 30, 'elevation': 25, 'azimuth': 45,
                    'zoom_factor': 1.6}
    colorbar_size = (800, 20)

    seismic_vol = np.load(seismic_path)
    semblance_vol = np.load(fault_path)
    seismic_vol = np.squeeze(seismic_vol)
    semblance_vol = np.squeeze(semblance_vol)
    semblance_vol = (semblance_vol > 0.5).astype(np.float32)
    seismic_cmap = 'seismic'
    vmax = np.max(np.abs(seismic_vol)) / 2
    seismic_range = (-vmax, vmax)
    colors = [(0, 0, 0, 0),
            (0, 0, 0, 1)]
    semblance_cmap = Colormap(colors)
    semblance_range = (0, 1.0)
    visual_nodes = volume_slices([seismic_vol, semblance_vol],
    cmaps=[seismic_cmap, semblance_cmap],
    clims=[seismic_range, semblance_range],
    interpolation='bilinear', **slicing)
    xyz_axis = XYZAxis()
    colorbar = Colorbar(cmap=seismic_cmap, clim=seismic_range,
                        label_str='Fault Semblance', size=colorbar_size)
    canvas2 = SeismicCanvas(title='Fault Semblance',
                            visual_nodes=visual_nodes,
                            xyz_axis=xyz_axis,
                            colorbar=colorbar,
                            **canvas_params)
    app.run()


def seismic3D(seismic_path):

    slicing = {'x_pos': 15, 'y_pos': 15, 'z_pos': 407}
    canvas_params = {'size': (900, 900),
                    'axis_scales': (0.5, 0.5, 0.5), # stretch z-axis
                    'colorbar_region_ratio': 0.1,
                    'fov': 30, 'elevation': 25, 'azimuth': 45,
                    'zoom_factor': 1.6}
    colorbar_size = (800, 20)

    seismic_vol = np.load(seismic_path)
    seismic_vol = np.squeeze(seismic_vol)
    seismic_cmap = 'seismic'
    vmax = np.max(np.abs(seismic_vol)) / 2
    seismic_range = (-vmax, vmax)
    visual_nodes = volume_slices([seismic_vol],
    cmaps=[seismic_cmap],
    clims=[seismic_range],
    interpolation='bilinear', **slicing)
    xyz_axis = XYZAxis()
    colorbar = Colorbar(cmap=seismic_cmap, clim=seismic_range,
                        label_str='Fault Semblance', size=colorbar_size)
    canvas2 = SeismicCanvas(title='Fault Semblance',
                            visual_nodes=visual_nodes,
                            xyz_axis=xyz_axis,
                            colorbar=colorbar,
                            **canvas_params)
    app.run()


if __name__ == '__main__':
    seismic_path = r"E:\conda\jupyter_notebook\ResACEUnet-master\fault_detection\datasets\test\seismic\Seismic_data.npy"
    fault_path = r"E:\conda\jupyter_notebook\ResACEUnet-master\fault_detection\datasets\test\fault\Seismic_nnunet.npy"
    seismic3D(seismic_path)
    fault3D(seismic_path,fault_path)
