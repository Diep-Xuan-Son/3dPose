# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import cv2
import math
import numpy as np
import paddle
import yaml

from det_keypoint_unite_utils import argsparser
from preprocess import decode_image
from infer import Detector, DetectorPicoDet, PredictConfig, print_arguments, get_test_images, bench_log
from keypoint_infer import KeyPointDetector, PredictConfig_KeyPoint
from visualize import visualize_pose
from benchmark_utils import PaddleInferBenchmark
from utils import get_current_memory_mb
from keypoint_postprocess import translate_to_ori_images

KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown'
}


def predict_with_given_det(image, det_res, keypoint_detector,
                           keypoint_batch_size, run_benchmark):
    rec_images, records, det_rects = keypoint_detector.get_person_from_rect(
        image, det_res)
    keypoint_vector = []
    score_vector = []

    rect_vector = det_rects
    keypoint_results = keypoint_detector.predict_image(
        rec_images, run_benchmark, repeats=10, visual=False)
    keypoint_vector, score_vector = translate_to_ori_images(keypoint_results,
                                                            np.array(records))
    keypoint_res = {}
    keypoint_res['keypoint'] = [
        keypoint_vector.tolist(), score_vector.tolist()
    ] if len(keypoint_vector) > 0 else [[], []]
    keypoint_res['bbox'] = rect_vector
    return keypoint_res


def topdown_unite_predict(detector,
                          topdown_keypoint_detector,
                          image_list,
                          keypoint_batch_size=1,
                          save_res=False):
    det_timer = detector.get_timer()
    store_res = []
    for i, img_file in enumerate(image_list):
        # Decode image in advance in det + pose prediction
        det_timer.preprocess_time_s.start()
        image, _ = decode_image(img_file, {})
        det_timer.preprocess_time_s.end()

        if FLAGS.run_benchmark:
            results = detector.predict_image(
                [image], run_benchmark=True, repeats=10)

            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
        else:
            results = detector.predict_image([image], visual=False)
        results = detector.filter_box(results, FLAGS.det_threshold)
        if results['boxes_num'] > 0:
            keypoint_res = predict_with_given_det(
                image, results, topdown_keypoint_detector, keypoint_batch_size,
                FLAGS.run_benchmark)

            if save_res:
                save_name = img_file if isinstance(img_file, str) else i
                store_res.append([
                    save_name, keypoint_res['bbox'],
                    [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
                ])
        else:
            results["keypoint"] = [[], []]
            keypoint_res = results
        if FLAGS.run_benchmark:
            cm, gm, gu = get_current_memory_mb()
            topdown_keypoint_detector.cpu_mem += cm
            topdown_keypoint_detector.gpu_mem += gm
            topdown_keypoint_detector.gpu_util += gu
        else:
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            visualize_pose(
                img_file,
                keypoint_res,
                visual_thresh=FLAGS.keypoint_threshold,
                save_dir=FLAGS.output_dir)
    if save_res:
        """
        1) store_res: a list of image_data
        2) image_data: [imageid, rects, [keypoints, scores]]
        3) rects: list of rect [xmin, ymin, xmax, ymax]
        4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
        5) scores: mean of all joint conf
        """
        with open("det_keypoint_unite_image_results.json", 'w') as wf:
            json.dump(store_res, wf, indent=4)


def topdown_unite_predict_video(detector,
                                topdown_keypoint_detector,
                                camera_id,
                                keypoint_batch_size=1,
                                save_res=False):
    for file_video in sorted(os.listdir(FLAGS.video_file)):
        print(file_video)
        video_name = 'output.mp4'
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(os.path.join(FLAGS.video_file, file_video))
            video_name = os.path.split(os.path.join(FLAGS.video_file, file_video))[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        #-----------------------------------------------------------
        event_names = {
            0: 'Address',
            1: 'Toe-up',
            2: 'Mid-backswing',
            3: 'Top',
            4: 'Mid-downswing',
            5: 'Impact',
            6: 'Mid-follow-through',
            7: 'Finish',
            8: 'None'
        }
        for i in range(len(event_names)):
            if not os.path.isdir("./annotations/{}".format(event_names[i])):
                os.mkdir("./annotations/{}".format(event_names[i]))
        # event_path = 
        name_video = file_video.split(".")[0]
        json_data = json.load(open("label_golf.json"))
        frame_label = np.array(json_data[name_video])
        frame_label -= frame_label[0]
        print(frame_label)
        #-----------------------------------------------------------

        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        out_path = os.path.join(FLAGS.output_dir, video_name)
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        index = 0
        store_res = []
        idx_label = 1 #them
        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            index += 1
            print('detect frame: %d' % (index))
            #--------------------------
            if(frame_label[idx_label] < index):
                if save_res:
                    """
                    1) store_res: a list of frame_data
                    2) frame_data: [frameid, rects, [keypoints, scores]]
                    3) rects: list of rect [xmin, ymin, xmax, ymax]
                    4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
                    5) scores: mean of all joint conf
                    """
                    num_file = len(os.listdir("./annotations/{}".format(event_names[idx_label-1])))
                    with open("./annotations/{}/{}.json".format(event_names[idx_label-1], event_names[idx_label-1] + "_" + name_video + "_" + str(num_file)), 'w') as wf:
                        json.dump(store_res, wf, indent=4)
                idx_label += 1
                # store_res = []

            #--------------------------

            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = detector.predict_image([frame2], visual=False)
            results = detector.filter_box(results, FLAGS.det_threshold)
            if results['boxes_num'] == 0:
                writer.write(frame)
                continue

            keypoint_res = predict_with_given_det(
                frame2, results, topdown_keypoint_detector, keypoint_batch_size,
                FLAGS.run_benchmark)

            im = visualize_pose(
                frame,
                keypoint_res,
                visual_thresh=FLAGS.keypoint_threshold,
                returnimg=True)
            if save_res:
                if(len(keypoint_res['keypoint'][1]) > 1):
                    kpt_score_max = [0]
                    idx_max = 0
                    for idx, kpt in enumerate(keypoint_res['keypoint'][1]):
                        if(kpt>kpt_score_max):
                            kpt_score_max = kpt
                            idx_max = idx
                    keypoint_res['keypoint'][1] = kpt_score_max
                    keypoint_res['bbox'] = keypoint_res['bbox'][idx_max]
                    keypoint_res['keypoint'][0] = keypoint_res['keypoint'][0][idx_max]

                store_res.append([
                    name_video + "_" + str(index), keypoint_res['bbox'],
                    [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
                ])
                if(len(store_res)>1):
                    store_res = store_res[1:]

            writer.write(im)
            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        writer.release()
        print('output_video saved to: {}'.format(out_path))
        # if save_res:
        #     """
        #     1) store_res: a list of frame_data
        #     2) frame_data: [frameid, rects, [keypoints, scores]]
        #     3) rects: list of rect [xmin, ymin, xmax, ymax]
        #     4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
        #     5) scores: mean of all joint conf
        #     """
        #     with open("./annotations//det_keypoint_unite_video_results.json", 'w') as wf:
        #         json.dump(store_res, wf, indent=4)


def main():
    deploy_file = os.path.join(FLAGS.det_model_dir, 'infer_cfg.yml')
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)
    arch = yml_conf['arch']
    detector_func = 'Detector'
    if arch == 'PicoDet':
        detector_func = 'DetectorPicoDet'

    detector = eval(detector_func)(FLAGS.det_model_dir,
                                   device=FLAGS.device,
                                   run_mode=FLAGS.run_mode,
                                   trt_min_shape=FLAGS.trt_min_shape,
                                   trt_max_shape=FLAGS.trt_max_shape,
                                   trt_opt_shape=FLAGS.trt_opt_shape,
                                   trt_calib_mode=FLAGS.trt_calib_mode,
                                   cpu_threads=FLAGS.cpu_threads,
                                   enable_mkldnn=FLAGS.enable_mkldnn,
                                   threshold=FLAGS.det_threshold)

    topdown_keypoint_detector = KeyPointDetector(
        FLAGS.keypoint_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.keypoint_batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        use_dark=FLAGS.use_dark)
    keypoint_arch = topdown_keypoint_detector.pred_config.arch
    assert KEYPOINT_SUPPORT_MODELS[
        keypoint_arch] == 'keypoint_topdown', 'Detection-Keypoint unite inference only supports topdown models.'

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        topdown_unite_predict_video(detector, topdown_keypoint_detector,
                                    FLAGS.camera_id, FLAGS.keypoint_batch_size,
                                    FLAGS.save_res)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        topdown_unite_predict(detector, topdown_keypoint_detector, img_list,
                              FLAGS.keypoint_batch_size, FLAGS.save_res)
        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
            topdown_keypoint_detector.det_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            det_model_dir = FLAGS.det_model_dir
            det_model_info = {
                'model_name': det_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, det_model_info, name='Det')
            keypoint_model_dir = FLAGS.keypoint_model_dir
            keypoint_model_info = {
                'model_name': keypoint_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(topdown_keypoint_detector, img_list, keypoint_model_info,
                      FLAGS.keypoint_batch_size, 'KeyPoint')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()

