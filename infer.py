import pyrealsense2 as rs
import numpy as np
import cv2


ctx = rs.context()
serials = []
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()

if len(ctx.devices) > 0:
    for dev in ctx.devices:
        print('Found device:', dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
        serials.append(dev.get_info(rs.camera_info.serial_number))
else:
    print("No Intel Device connected")

pipelines = []
windows = []

for serial in serials:
    pipe = rs.pipeline(ctx)
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    pipelines.append(pipe)

    window_name = f"Camera {serial}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    windows.append(window_name)

try:
    while True:
        for pipe, window_name in zip(pipelines, windows):
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

            cv2.imshow(window_name, color_image)
            cv2.imshow(window_name + " Depth", depth_colormap)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

finally:
    for pipe in pipelines:
        pipe.stop()

    cv2.destroyAllWindows()