import torch
import numpy as np
from gastnet.model.gast_net import SpatioTemporalModel
from gastnet.tools.preprocess import load_kpts_json, revise_skes
from gastnet.tools.inference import gen_pose
from gastnet.tools.vis_h36m import render_animation
from gastnet.common.skeleton import Skeleton
from gastnet.common.graph_utils import adj_mx_from_skeleton
from gastnet.tools.color_edge import h36m_color_edge, custom_color_edge
# from gastnet.common.camera import world_to_camera, image_coordinates
import matplotlib.pyplot as plt
import time
import cv2

skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
            joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
            joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
adj = adj_mx_from_skeleton(skeleton)
joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
keypoints_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M', 'num_joints': 17}

rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
# parents = np.expand_dims(np.array(skeleton.parents()), axis=1)
# index = np.expand_dims(np.array([i for i in np.arange(17)]), axis=1)
# edges = np.concatenate((index, parents), axis=1)

def load_model_layer(model_dir, rf=27):
    if rf == 27:
        chk = model_dir + '27_frame_model.bin'
        filters_width = [3, 3, 3]
        channels = 128
    elif rf == 81:
        chk = model_dir + '81_frame_model.bin'
        filters_width = [3, 3, 3, 3]
        channels = 64
    else:
        raise ValueError('Only support 27 and 81 receptive field models for inference!')

    print('Loading GAST-Net ...')
    model_pos = SpatioTemporalModel(adj, 17, 2, 17, filter_widths=filters_width, channels=channels, dropout=0.05)

    # Loading pre-trained model
    checkpoint = torch.load(chk)
    model_pos.load_state_dict(checkpoint['model_pos'])

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
    model_pos = model_pos.eval()

    print('GAST-Net successfully loaded')

    return model_pos

def visualize_image(im, keypoints, poses, azim, output, viewport, limit=-1,
                     downsample=1, size=5):
    num_person = keypoints.shape[1]
    
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)

    ax_in.imshow(im)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()

    ax_3d = []
    lines_3d = []
    radius = 1.7
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
        # points_3d.append([])#them
    poses = list(poses.values())

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

    i=0
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

        for n, ax in enumerate(ax_3d):
            pos = poses[n][i]
            lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                       [pos[j, 1], pos[j_parent, 1]],
                                       [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, linewidth=4))
            ax.plot3D([pos[j, 0]], [pos[j, 1]], [pos[j, 2]], c='red', marker="o", markersize=5, zorder=10, markerfacecolor='white')
    points = ax_in.scatter(*keypoints[i].reshape(17*num_person, 2).T, 25, color=colors_2d, edgecolors='white', zorder=10)

    fig.tight_layout()
    #----
    # fig.canvas.draw()
    # img_w, img_h = fig.canvas.get_width_height()
    # img_vis = np.frombuffer(
    #     fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)
    # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("abc.jpg", img_vis)    
    #----
    plt.savefig(output)

def Gen_3D(re_kpts, valid_frames, width, height, model_pos, num_person, video_file, rf=27):
    pad = (rf - 1) // 2  # Padding on each side
    causal_shift = 0

    # Generating 3D poses
    prediction = gen_pose(re_kpts, valid_frames, width, height, model_pos, pad, causal_shift)
    # print(prediction)
    # exit()

    ab_dis = False
    # Adding absolute distance to 3D poses and rebase the height
    if num_person == 2:
        prediction = revise_skes(prediction, re_kpts, valid_frames)
    elif ab_dis:
        prediction[0][:, :, 2] -= np.expand_dims(np.amin(prediction[0][:, :, 2], axis=1), axis=1).repeat([17], axis=1)
    else:
        prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])

    # If output two 3D human poses, put them in the same 3D coordinate system
    same_coord = False
    if num_person == 2:
        same_coord = True
    anim_output = {}
    for i, anim_prediction in enumerate(prediction):
        anim_output.update({'Reconstruction %d' % (i+1): anim_prediction})

    viz_output = './output3d/' + 'animation_' + video_file.split('/')[-1].split('.')[0] + '.mp4'
    print('Generating animation ...')
    # re_kpts: (M, T, N, 2) --> (T, M, N, 2)

    re_kpts = re_kpts.transpose(1, 0, 2, 3)
    # print(np.array(list(anim_output.values())))
    # print(np.array(list(anim_output.values())).shape)
    # t1 = time.time()
    render_animation(re_kpts, keypoints_metadata, anim_output, skeleton, 25, 30000, np.array(70., dtype=np.float32),
                     viz_output, input_video_path=video_file, viewport=(width, height), com_reconstrcution=same_coord)
    # print("Render 3D time: ", time.time()-t1)
    # render_animation_test(re_kpts, keypoints_metadata, anim_output, skeleton, 25, np.array(70., dtype=np.float32),
    #                  viz_output, input_video_path=video_file, viewport=(width, height))


def Gen_3D_image(re_kpts, valid_frames, im, model_pos, num_person, path_event, rf=27):
    height = im.shape[0]
    width = im.shape[1]
    # print(im)

    pad = (rf - 1) // 2  # Padding on each side
    causal_shift = 0

    # Generating 3D poses
    prediction = gen_pose(re_kpts, valid_frames, width, height, model_pos, pad, causal_shift)
    # print(prediction)
    # exit()

    ab_dis = False
    # Adding absolute distance to 3D poses and rebase the height
    if num_person == 2:
        prediction = revise_skes(prediction, re_kpts, valid_frames)
    elif ab_dis:
        prediction[0][:, :, 2] -= np.expand_dims(np.amin(prediction[0][:, :, 2], axis=1), axis=1).repeat([17], axis=1)
    else:
        prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])

    re_kpts = re_kpts.transpose(1, 0, 2, 3)
    img_output = path_event + "_3D.jpg"

    anim_output = {}
    for i, anim_prediction in enumerate(prediction):
        anim_output.update({'Reconstruction %d' % (i+1): anim_prediction})
    
    #---- test other draw ------------------
    # prediction_to_camera = []
    # for i in range(len(prediction)):
    #     sub_prediction = prediction[i]

    #     sub_prediction = world_to_camera(sub_prediction, R=rot, t=0)

    #     # sub_prediction[:, :, 2] -= np.expand_dims(np.amin(sub_prediction[:, :, 2], axis=1), axis=1).repeat([17], axis=1)
    #     # sub_prediction[:, :, 2] -= np.amin(sub_prediction[:, :, 2])

    #     prediction_to_camera.append(sub_prediction)
    # pose3d = prediction_to_camera
    # pose3d = np.array(list(pose3d))
    # print(pose3d)

    # parents = skeleton.parents()
    # index = [i for i in np.arange(17)]

    # img = np.zeros((width, width, 3), dtype='uint8')
    # img[:] = (255,255,255)

    # for j, j_parent in zip(index, parents):
    #   if j_parent == -1:
    #       continue
    #   if len(parents) == 17 and keypoints_metadata['layout_name'] != 'coco':
    #       print((pose3d[0,0,j,0], pose3d[0,0,j,1]))
    #       print((re_kpts[0,0,j,0], re_kpts[0,0,j,1]))
    #       image = cv2.line(img, (pose3d[0,0,j,0], pose3d[0,0,j,1]), (pose3d[0,0,j_parent,1], pose3d[0,0,j_parent,1]), (255,0,0), 9)
    
    # cv2.imsave("test.jpg", image)
    # exit()
    #---------------------
    visualize_image(im, re_kpts, anim_output, np.array(70., dtype=np.float32),
                     img_output, viewport=(width, height))
    return prediction