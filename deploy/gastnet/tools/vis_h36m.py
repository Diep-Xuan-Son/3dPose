import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
from gastnet.tools.color_edge import h36m_color_edge, custom_color_edge

import cv2


def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)


def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-vcodec', 'rawvideo', '-']

    i = 0
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)

def render_animation_test(keypoints, keypoints_metadata, poses, skeleton, fps, azim, output, viewport, limit=-1,
                     downsample=1, size=5, input_video_path=None):
    print("Visualize 3D...")
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out3d = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    poses = np.array(list(poses.values()))
    i = 0
    while(1):
        ret, frame = cap.read()
        if not ret:
            break

        num_person = keypoints.shape[1]
        
        fig = plt.figure(figsize=(size * (1 + len(poses)), size))
        ax_in = fig.add_subplot(1, 2, 1)

        ax_in.imshow(frame)
        ax_in.get_xaxis().set_visible(False)
        ax_in.get_yaxis().set_visible(False)
        ax_in.set_axis_off()

        ax_3d = []
        lines_3d = []
        radius = 1.7

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.dist = 7.5


        # for index, (title, data) in enumerate(poses.items()):
        #     ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')

        #     ax.view_init(elev=15., azim=azim)
        #     ax.set_xlim3d([-radius / 2, radius / 2])
        #     ax.set_zlim3d([0, radius])
        #     ax.set_ylim3d([-radius / 2, radius / 2])
        #     # ax.set_aspect('equal')
        #     ax.set_xticklabels([])
        #     ax.set_yticklabels([])
        #     ax.set_zticklabels([])
        #     ax.dist = 7.5
        #     # ax.set_title(title)  # , pad=35
        #     ax_3d.append(ax)
        #     lines_3d.append([])
        #     # points_3d.append([])#them

        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')

        initialized = False
        image = None
        lines = []
        points = None

        limit = len(all_frames)

        parents = skeleton.parents()
        index = [i for i in np.arange(17)]

        joints_left_2d = keypoints_metadata['keypoints_symmetry'][0]#them
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]

        colors_2d = np.full(17, 'black')
        colors_2d[joints_right_2d] = 'red'
        colors_2d[joints_left_2d] = 'red'   #them

        for j, j_parent in zip(index, parents):
            if j_parent == -1:
                continue
            # print(keypoints)
            # print(keypoints.shape)
            if len(parents) == 17 and keypoints_metadata['layout_name'] != 'coco':
                # for m in range(num_person):
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                ax_in.plot([keypoints[i, 0, j, 0], keypoints[i, 0, j_parent, 0]],
                            [keypoints[i, 0, j, 1], keypoints[i, 0, j_parent, 1]],
                            color='pink', linewidth=4)
                ax_in.plot(keypoints[i, 0, j, 0], keypoints[i, 0, j, 1], c='red', marker="o", markersize=5, zorder=10, markerfacecolor='white')

            # Apply different colors for each joint
            # col = h36m_color_edge(j)
            col = custom_color_edge(j)

            # print(poses)
            # print(poses.shape)
            ax.plot([poses[0, i, j, 0], poses[0, i, j_parent, 0]], 
                    [poses[0, i, j, 1], poses[0, i, j_parent, 1]],
                    [poses[0, i, j, 2], poses[0, i, j_parent, 2]], zdir='z', c=col, linewidth=4)
            ax.plot3D([poses[0, i, j, 0]], [poses[0, i, j, 1]], [poses[0, i, j, 2]], c='red', marker="o", markersize=5, zorder=10, markerfacecolor='white')
        # points = ax_in.scatter(*keypoints[i].reshape(17*num_person, 2).T, 25, color=colors_2d, edgecolors='white', zorder=10)

        fig.tight_layout()
        #----
        fig.canvas.draw()
        img_w, img_h = fig.canvas.get_width_height()
        img_vis = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite("abc.jpg", img_vis)    
        # out3d.write(img_vis)
        #----
        # plt.savefig(output)
    out3d.release()
    print("Done!")

def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport, limit=-1,
                     downsample=1, size=5, input_video_path=None, com_reconstrcution=False, input_video_skip=0):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    # plt.ioff()
    # print(keypoints)
    # print(poses)

    num_person = keypoints.shape[1]
    if num_person == 2 and com_reconstrcution:

        fig = plt.figure(figsize=(size * (1 + len(poses)), size))
        ax_in = fig.add_subplot(1, 2, 1)
    else:
        fig = plt.figure(figsize=(size * (1 + len(poses)), size))
        ax_in = fig.add_subplot(1, 1 + len(poses), 1)

    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    # ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    radius = 1.7
    points_3d = [] #them

    if num_person == 2 and com_reconstrcution:
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius, radius])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius, radius])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax_3d.append(ax)
        lines_3d.append([])

        poses = list(poses.values())
    else:
        for index, (title, data) in enumerate(poses.items()):
            ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')

            ax.view_init(elev=15., azim=azim)
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_zlim3d([0, radius])
            ax.set_ylim3d([-radius / 2, radius / 2])
            # ax.set_aspect('equal')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 7.5
            # ax.set_title(title)  # , pad=35
            ax_3d.append(ax)
            lines_3d.append([])
            points_3d.append([])#them
        poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

        keypoints = keypoints[input_video_skip:]  # todo remove
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]

        if fps is None:
            fps = get_fps(input_video_path)

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    index = [i for i in np.arange(17)]

    def update_video(i):
        nonlocal initialized, image, lines, points

        joints_left_2d = keypoints_metadata['keypoints_symmetry'][0]#them
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]

        if num_person == 2:
            joints_right_2d_two = []
            joints_right_2d_two += joints_right_2d
            joints_right_2d_second = [i + 17 for i in joints_right_2d]
            joints_right_2d_two += joints_right_2d_second

            colors_2d = np.full(34, 'black')
            colors_2d[joints_right_2d_two] = 'red'
        else:
            colors_2d = np.full(17, 'black')
            colors_2d[joints_right_2d] = 'red'
            colors_2d[joints_left_2d] = 'red'   #them

        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')

            for j, j_parent in zip(index, parents):
                if j_parent == -1:
                    continue

                if len(parents) == 17 and keypoints_metadata['layout_name'] != 'coco':
                    for m in range(num_person):
                        # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                        lines.append(ax_in.plot([keypoints[i, m, j, 0], keypoints[i, m, j_parent, 0]],
                                                [keypoints[i, m, j, 1], keypoints[i, m, j_parent, 1]],
                                                color='pink', linewidth=4))

                # Apply different colors for each joint
                # col = h36m_color_edge(j)
                col = custom_color_edge(j)

                if com_reconstrcution:
                    for pose in poses:
                        pos = pose[i]
                        lines_3d[0].append(ax_3d[0].plot([pos[j, 0], pos[j_parent, 0]],
                                                         [pos[j, 1], pos[j_parent, 1]],
                                                         [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, linewidth=3))
                        # points_3d[0].append(ax_3d[0].plot([pos[j, 0]], [pos[j, 1]], [pos[j, 2]], zdir='z', c="blue", z=3,  marker = "o"))#them
                else:
                    for n, ax in enumerate(ax_3d):

                        pos = poses[n][i]
                        # print(lines_3d[n])
                        lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                   [pos[j, 1], pos[j_parent, 1]],
                                                   [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, linewidth=4))
                        points_3d[n].append(ax.plot3D([pos[j, 0]], [pos[j, 1]], [pos[j, 2]], c='red', marker="o", markersize=5, zorder=10, markerfacecolor='white'))
            
                        # points_3d[n].set_data (pos[j, 0], pos[j, 1])
                        # points_3d[n].set_3d_properties(pos[j, 2])
            points = ax_in.scatter(*keypoints[i].reshape(17*num_person, 2).T, 25, color=colors_2d, edgecolors='white', zorder=10)
            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in zip(index, parents):
                if j_parent == -1:
                    continue

                if len(parents) == 17 and keypoints_metadata['layout_name'] != 'coco':
                    for m in range(num_person):
                        lines[j + 16*m - 1][0].set_data([keypoints[i, m, j, 0], keypoints[i, m, j_parent, 0]],
                                                        [keypoints[i, m, j, 1], keypoints[i, m, j_parent, 1]])

                if com_reconstrcution:
                    for k, pose in enumerate(poses):
                        pos = pose[i]
                        lines_3d[0][j + k*16 - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                        lines_3d[0][j + k*16 - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                        lines_3d[0][j + k*16 - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
                       
                else:
                    for n, ax in enumerate(ax_3d):
                        pos = poses[n][i]
                        # print(lines_3d[n][j - 1])
                        lines_3d[n][j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                        lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                        lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
                        # --------------------------
                        points_3d[n][j - 1][0].set_xdata([pos[j, 0]])
                        points_3d[n][j - 1][0].set_ydata([pos[j, 1]])
                        points_3d[n][j - 1][0].set_3d_properties([pos[j, 2]], zdir='z')
            # points_3d1._offsets3d = (pos[j, 0], pos[j, 1], pos[j, 2])
            # print(keypoints[i])
            points.set_offsets(keypoints[i].reshape(17*num_person, 2))

        print('{}/{}      '.format(i, limit), end='\r')

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        # Writer = writers['imagemagick']
        # writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        # anim.save(output, writer=writer)
        anim.save(output, dpi=80, writer=ImageMagickWriter())
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()
