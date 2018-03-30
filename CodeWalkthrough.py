tools/train_net.py

imports []

--> main():
		
		# Initialize Caffe2
		workspace.GlobalInit(['caffe2', '--caffe2_log_level=0', '--caffe2_gpu_memory_tracking=1'])

		logger = setup_logging(__name__)
		--> $DETECTRON/lib/utils/logging.py
			logging.basicConfig(level=…)
			--> Python 2.7 logging libs
				...
			...
			logger = logging.getLogger(name)
				--> Python 2.7 logging libs
					...
				<--	return Logger.manager.getLogger(name)
		<--	return logger

		logging.getLogger('roi_data.loader').setLevel(logging.INFO)
		--> Python 2.7 logging libs
			...
		<--	None

		args = parse_args()
		--> same file
			... Parse all arguments like config-file, skip test, multi-gpu test, options.
		<--	return parser.parse_args()

		logger.info('Called with args:')
		--> Python 2.7 logging libs
			...
		<--	None

		logger.info(args)
		--> Python 2.7 logging libs
			...
		<--	None

		if args.cfg_file is not None: True
			merge_cfg_from_file(args.cfg_file)
			--> $DETECTRON/lib/core/config.py
				open yaml file: f
				yaml_cfg = AttrDict(yaml.load(f))
				--> Python 2.7 yam libs
				<-- return dictionary  with model (hyper)paramters
					{'FAST RCNN': {'ROI...'}...}

		if args.opts is not None: True
			merge_cfg_from_list(cfg_filename=args.opts)
			--> $DETECTRON/lib/core/config.py
				if _key_is_deprecated(full_key): False
				f _key_is_renamed(full_key): False
				...
				...
				# Processes paramters dependent on config file. Like output directory.
			<--	None

		assert_and_infer_cfg()
		 """Call this function in your script after you have finished setting all cfg
    		values that are necessary (e.g., merging a config from a file, merging
		    command line config options, etc.). By default, this function will also
		    mark the global cfg as immutable to prevent changing the global cfg settings
		    during script execution (which can lead to hard to debug errors or code
		    that's harder to understand than is necessary).
   		"""
		--> $DETECTRON/lib/core/config.py
			if _C.MODEL.RPN_ONLY or _C.MODEL.FASTER_RCNN: True
				_C.RPN.RPN_ON = True
			if _C.RPN.RPN_ON or _C.RETINANET.RETINANET_ON: True
				_C.TEST.PRECOMPUTED_PROPOSALS = False
			if cache_urls: True
				cache_cfg_urls()
				--> 
				"""
				Download URLs in the config, cache them locally, and rewrite cfg to make
				use of the locally cached file.
				"""
			<-- None
		<--	None

		logger.info('Training with config:')
		--> Python 2.7 logging libs
			...
		<--	None
		logger.info(pprint.pformat(cfg)) # Pretty print

		np.random.seed(cfg)
		--> $DETECTRON/lib/utils/collections.py
		<-- return self[name]:3

		checkpoints = train_model()
		--> same file
			logger = logging.getLogger(__name__)
			--> Python 2.7 logging libs
			...
			<--	return Logger.manager.getLogger(name)

			model, weights_file, start_iter, checkpoints, output_dir = create_model()
			--> same file
				logger = ...
				start_iter = 0
				checkpoints = {}
				output_dir = get_output_dir(training=True)
				--> $DETECTRON/lib/core/config.py
					dataset = __C.TRAIN.DATASETS if training else __C.TEST.DATASETS
				<--	return outdir:/tmp/..generalized_rcnn
				if cfg.TRAIN.AUTO_RESUME: True
					final_path = os.path.join(output_dir, 'model_final.pkl')
					check if final_path exits
						if True: 
							no need to training
							return final_path, output_dir
					for file in output_dir files:
						find the latest saved model:
							checkpoint_iter = latest saved model iter
							start_iter = checkpoint_iter + 1
							resume_weights = file_var of latest saved model
					if start_iter > 0: False
						# Override random weight initialization with weights from a saved model
						...
				logger.info('Building model: {}'.format(cfg.MODEL.TYPE))
				--> $DETECTRON/lib/utils/collections.py
				<-- return {'FASTER_RCNN...CLASSES':81}
				--> $DETECTRON/lib/utils/collections.py
				<-- return TYPE='generalized_rcnn'
				--> Python 2.7 logging libs
					...
				<--	None
				model = model_builder.create(cfg.MODEL.TYPE, train=True)
				--> $DETECTRON/lib/utils/collections.py
				<-- return {'FASTER_RCNN...CLASSES':81}
				--> $DETECTRON/lib/utils/collections.py
				<-- return TYPE='generalized_rcnn'
				--> $DETECTRON/lib/modeling/model_builder.py
					model = DetectionModelHelper(name=model_type_func, 
												train=train, 
												num_classes=cfg.MODEL.NUM_CLASSES)
					<-- return {'FASTER_RCNN...CLASSES':81}
					--> $DETECTRON/lib/utils/collections.py
					<-- return 81
					init_params=train
					model.only_build_forward_pass = False
					model.target_gpu_id = gpu_id
					return get_func(model_type_func)(model)
							--> get_func(func_name)
							<-- return <function 34f...>
					--> generalized_rcnn(model):
					<-- return build_generic_detection_model(model, get_func(cfg.MODEL.CONV_BODY))
															--> $DETECTRON/lib/utils/collections.py
															<-- return 'ResNet.add_R...50_conv4_body'
						...
						add_roi_box_head_func=...
						add_roi_mask_head_func=...
						add_roi_keypoint_head_func=...
						freeze_conv_body=cfg.TRAIN.FREEZE_CONV_BODY
				optimize_memory(model):
				--> same file
					for device in range(cfg.NUM_GPUS):
						name_of_gpu = ...
						losses = ...
						model.net._net = memonger.share_grad_blobs(model.net losses, set(model.param_to_grad.values()))
																					--> ../caffe2/python/core.py
																						__hash__():
																					<-- return hash(self._name)
						workspace.RunNetOnce(model.param_init_net)
						--> ../caffe2/python/workspace.py
							<-- return CallWithExceptionIntercept(C.run_net_once, 
																C.Workspace.current._last_failed_op_net_position,
																GetNetName(net))
																--> GetNetName(net)
																<-- return net.Name():u'generalized_rcnn_init'

							--> StringifyProto(net)
								"""
								Stringify a protocol buffer object.
								  Inputs:
								    obj: a protocol buffer object, or a Pycaffe2 object that has a Proto()
								        function.
								  Outputs:
								    string: the output protobuf string.
								  Raises:
								    AttributeError: if the passed in object does not have the right attribute.
								"""
							<-- return some string
							--> CallWithExceptionIntercept(func, op_id_fetcher, net_name, *args, **kwargs)
							<-- return True
						<-- return True 	
		<-- return model, start_iter, checkpoints, output_dir

		if 'final' in checkpoints: False

		setup_model_for_training(model, output_dir)
		--> same file
			logger = logging.getLogger(...)

			add_model_training_inputs(model)
			--> logger = ...
				logger.info('Loading dataset: {}'.format(cfg.TRAIN.DATASETS))
				roidb = combined_roidb_for_training(cfg.TRAIN.DATASETS, 
													cfg.TRAIN.PROPOSAL_FILES)
				--> $DETECTRON/lib/utils/collections.py
				<-- Loads annotations, training and cross-validation datasets(augmented)
							Computes bb targets
				logger.info('{:d} roidb entries'.format(len(roidb)))
				--> Python 2.7 libs
				<-- None
				model_builder.add_training_inputs(model, roidb=roidb)
				--> $DETECTRON/lib/modeling/model_builder.py
					assert mode.train, 'Training inputs can only be added to training model'
					if roidb is not None:
						model.roi_data_loader = RoIDataLoader(roidb, num_loaders=cfg.DATA_LOADER.NUM_THREADS)
						--> $DETECTRON/lib/utils/collections.py
						<-- return {'NUM_THREADS': 4}
						
						--> $DETECTRON/lib/utils/collections.py
						<-- return 4
						
						--> $DETECTRON/lib/roi_data/loader.py
						<-- return None


