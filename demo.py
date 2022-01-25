# Lines In Space demo
import depthai as dai
import numpy as np
import pygametoolkit
import time
import threading


class LISDemo:

    def __init__(self,
                 input_shape_detection=(640, 640),
                 input_shape_estimation=(256, 256),
                 input_shape_lifting=(256, 256)
                 ):
        '''

        :param input_shape_detection:
        :param input_shape_estimation:
        :param input_shape_estimation:
        '''
        self._input_shape_detection = input_shape_detection
        self._input_shape_estimation = input_shape_estimation
        self.path_to_detector = './blobs/demo/yolox_tiny_8x8_300e_coco_openvino_2021.4_3shave.blob'                 ##
        self.path_to_detector_post = './blobs/demo/decode_bb_openvino_2021.4_1shave.blob'                           ##
        self.path_to_estimator = './blobs/demo/hrnet_w32_mpii_256x256_openvino_2021.4_3shave.blob'                  ##
        self.path_to_estimator_post = './blobs/demo/decode_heatmaps_openvino_2021.4_1shave.blob'                    ##
        self.path_to_lifter = './blobs/demo/videopose_poseaug_openvino_2021.4_3shave.blob'                          ##
        self._path_to_div = './blobs/demo/DivideBy255_openvino_2021.4_1shave.blob'                                  ##

        self._pipeline = self._setup_pipeline()
        self._det_doi_bb = []
        self._game = pygametoolkit.GameDemo()

    def run(self):
        x_1 = threading.Thread(target=self._live_run)
        x_1.start()
        self._game.run()

    def _live_run(self):
        # Connect to device and start pipeline
        with dai.Device(self._pipeline) as device:

            device.setLogLevel(dai.LogLevel.WARN)
            device.setLogOutputLevel(dai.LogLevel.WARN)

            print('Usb speed: ', device.getUsbSpeed().name)

            self.q_lifter_out = device.getOutputQueue("nn_out_2", maxSize=1, blocking=False)
            self.q_bb_out = device.getOutputQueue("bb_out", maxSize=1, blocking=False)

            worker_opt = True
            while worker_opt:
                time_probe = time.time()

                bb_outq = self.q_bb_out.get()
                if bb_outq is not None:
                    bb_ou = bb_outq.getLayerFp16('input_1')
                    if bb_ou[4] > 0.85:
                        lifting_outq = self.q_lifter_out.get()
                        if lifting_outq is not None:
                            lifting_out = lifting_outq.getLayerFp16('output')
                            lifting_out = np.array(lifting_out, dtype=np.float32).reshape((1, 16, 3))
                            self._game.draw_pose(lifting_out)
                    else:
                        self._game._update_pose_test()
                worker_opt = self._game.worker_opt
                print('t:', time.time() - time_probe)

    def _setup_pipeline(self):
        l_pipeline = dai.Pipeline()
        l_pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)
        nrInferenceThreads_01 = 1       # detector decode
        nrPerThreads_01 = 1
        nrInferenceThreads_02 = 1       # div by 255
        nrPerThreads_02 = 1
        nrInferenceThreads_03 = 1       # pose decode
        nrPerThreads_03 = 1

        nrInferenceThreads_11 = 2       # detector
        nrPerThreads_11 = 1
        nrInferenceThreads_12 = 1       # estimator
        nrPerThreads_12 = 1
        nrInferenceThreads_13 = 1       # lifter
        nrPerThreads_13 = 1

        blockingopt_0 = False
        blockingopt_1 = True




        # setting up camera node
        camera_rgb = l_pipeline.create(dai.node.ColorCamera)
        camera_rgb.setPreviewSize(self._input_shape_detection[0], self._input_shape_detection[1])
        camera_rgb.setInterleaved(False)
        camera_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)  #

        # setting up person detector stage
        detector_nn = l_pipeline.create(dai.node.NeuralNetwork)
        detector_nn.setBlobPath(self.path_to_detector)
        detector_nn.setNumInferenceThreads(nrInferenceThreads_11)
        detector_nn.input.setBlocking(blockingopt_0)
        detector_nn.setNumNCEPerInferenceThread(nrPerThreads_11)


        # setting up person detector post processing stage
        detector_post_nn = l_pipeline.create(dai.node.NeuralNetwork)
        detector_post_nn.setBlobPath(self.path_to_detector_post)
        detector_post_nn.setNumInferenceThreads(nrInferenceThreads_01)
        detector_post_nn.input.setBlocking(blockingopt_1)
        detector_post_nn.setNumNCEPerInferenceThread(nrPerThreads_01)


        # setting up preprocess for estimation part I/III (crop) and II/III (resize)
        pre_0_manip = l_pipeline.create(dai.node.ImageManip)
        pre_0_manip.initialConfig.setCropRect(0., 0., 1., 1.)
        pre_0_manip.initialConfig.setResize(self._input_shape_estimation[0], self._input_shape_estimation[0])
        pre_0_manip.setMaxOutputFrameSize(self._input_shape_estimation[0]*self._input_shape_estimation[1]*3)
        pre_0_manip.inputImage.setQueueSize(1)
        pre_0_manip.inputImage.setBlocking(blockingopt_1)

        script_0 = l_pipeline.create(dai.node.Script)
        script_0.setScript("""
                        while True:                        
                            #node.warn("wally 0")

                            # pose landmarks from estimation
                            detection = node.io['test_1'].get().getLayerFp16('input_1')

                            xwidth = detection[2]-detection[0]                                                        
                            yheight = detection[3]-detection[1]
                            bsidehalf = max(xwidth, yheight) / 2 
                            xcenter = (detection[2] + detection[0]) / 2
                            ycenter = (detection[3] + detection[1]) / 2  
                            xmin = (xcenter - bsidehalf) / 640.0
                            xmin = max(0, xmin)
                            ymin = (ycenter - bsidehalf) / 640.0
                            ymin = max(0, ymin)
                            xmax = (xcenter + bsidehalf) / 640.0
                            xmax = min(1, xmax)
                            ymax = (ycenter + bsidehalf) / 640.0
                            ymax = min(1, ymax)
                            #node.warn(str(xmin)+','+str(ymin)+','+str(xmax)+','+str(ymax))
                            
                            cfg = ImageManipConfig()
                            
                            cfg.setCropRect(xmin, ymin, xmax, ymax)
                            cfg.setResize(256, 256)                            
                            node.io['to_manip_cfg'].send(cfg)
                """)

        # setting up preprocess for estimation part III/III (divide by 255)
        pre_2_manip = l_pipeline.create(dai.node.NeuralNetwork)
        pre_2_manip.setBlobPath(self._path_to_div)
        pre_2_manip.setNumInferenceThreads(nrInferenceThreads_02)
        pre_2_manip.input.setBlocking(blockingopt_1)
        pre_2_manip.setNumNCEPerInferenceThread(nrPerThreads_02)

        # setting up 2d pose landmark estimator stage
        estimator_nn = l_pipeline.create(dai.node.NeuralNetwork)
        estimator_nn.setBlobPath(self.path_to_estimator)
        estimator_nn.setNumInferenceThreads(nrInferenceThreads_12)
        estimator_nn.input.setBlocking(blockingopt_0)
        estimator_nn.setNumNCEPerInferenceThread(nrPerThreads_12)

        # setting up 2d pose landmark estimator stage
        estimator_post_nn = l_pipeline.create(dai.node.NeuralNetwork)
        estimator_post_nn.setBlobPath(self.path_to_estimator_post)
        estimator_post_nn.setNumInferenceThreads(nrInferenceThreads_03)
        estimator_post_nn.input.setBlocking(blockingopt_1)
        estimator_post_nn.setNumNCEPerInferenceThread(nrPerThreads_03)

        # setting up 3d pose lifter stage
        lifter_nn = l_pipeline.create(dai.node.NeuralNetwork)
        lifter_nn.setBlobPath(self.path_to_lifter)
        lifter_nn.setNumInferenceThreads(nrInferenceThreads_13)
        lifter_nn.input.setBlocking(blockingopt_0)
        lifter_nn.setNumNCEPerInferenceThread(nrPerThreads_13)

        # script
        script = l_pipeline.create(dai.node.Script)
        script.setScript("""
                        while True:                        
                            #node.warn("wally 1")

                            # pose landmarks from estimation
                            landmarks = node.io['test_1'].get().getLayerFp16('input_1')
                            

                            # bounding box from detection stage
                            value = node.io['test_0'].get().getLayerFp16('input_1')
                                               

                            bb = value[:4]
                            xwidth = bb[2]-bb[0]
                            yheight = bb[3]-bb[1]
                            bsidehalf = max(xwidth, yheight) / 2 
                            xcenter = (bb[2] + bb[0]) / 2
                            ycenter = (bb[3] + bb[1]) / 2  
                            xmin = (xcenter - bsidehalf) / 640.0
                            xmin = max(0, xmin)
                            ymin = (ycenter - bsidehalf) / 640.0
                            ymin = max(0, ymin)
                            xmax = (xcenter + bsidehalf) / 640.0
                            xmax = min(1, xmax)
                            ymax = (ycenter + bsidehalf) / 640.0
                            ymax = min(1, ymax)
                            bb = [xmin, ymin, xmax, ymax]   
                            
                            #bb = [ii / 640 for ii in bb]
                            bb = [ii * 256 for ii in bb]
                            
                            #node.warn(str(bb))           
                            size_img = 256, 256
                            size_bb = [bb[2]-bb[0], bb[3]-bb[1]]                  

                            # recover locations on the original image (from cropped space)
                            land_orig = [( ( xi / size_img[0] ) * size_bb[0] ) + bb[0] if not indx%2 else \
                                ( ( xi / size_img[1] ) * size_bb[1] ) + bb[1] for indx, xi in enumerate(landmarks)]

                            # normalize locations
                            land_norm = [land_i / size_img[0] * 2 -1 for land_i in land_orig]
                            #node.warn(str(land_norm))
                            
                            # send to lifter
                            lifting_nndata_in = NNData(len(land_norm))                            
                            lifting_nndata_in.setLayer("input_1", land_norm)
                            node.io['nndata'].send(lifting_nndata_in)

                """)

        #scripting_in = l_pipeline.create(dai.node.XLinkIn)
        #scripting_in.setStreamName("script_in")
        #scripting_in.out.link(script.inputs['test_0'])
        detector_post_nn.out.link(script.inputs['test_0'])
        estimator_post_nn.out.link(script.inputs['test_1'])

        # linking stuff
        camera_rgb.preview.link(detector_nn.input)
        camera_rgb.preview.link(pre_0_manip.inputImage)

        detector_nn.out.link(detector_post_nn.input)
        detector_post_nn.out.link(script_0.inputs['test_1'])

        pre_0_manip.out.link(pre_2_manip.input)
        pre_2_manip.out.link(estimator_nn.input)
        estimator_nn.out.link(estimator_post_nn.input)

        script_0.outputs['to_manip_cfg'].link(pre_0_manip.inputConfig)
        script.outputs['nndata'].link(lifter_nn.input)

        xout_lifter = l_pipeline.create(dai.node.XLinkOut)
        xout_lifter.setStreamName("nn_out_2")

        lifter_nn.out.link(xout_lifter.input)

        #--------------
        #xout_rgb = l_pipeline.create(dai.node.XLinkOut)
        #xout_rgb.setStreamName("rgb_out")
        #pre_0_manip.out.link(xout_rgb.input)
        xout_bb = l_pipeline.create(dai.node.XLinkOut)
        xout_bb.setStreamName("bb_out")
        detector_post_nn.out.link(xout_bb.input)

        return l_pipeline

