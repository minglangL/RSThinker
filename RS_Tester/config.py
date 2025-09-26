import os.path
import warnings

from RS_Tester.metrics import *

all_test_dataset_dict = {
    # SC
    "SC_AID": {
        'question_path': "sc/AID/sample_AID_question.json",
        'image_path': "sc/AID/",
        'image_key': 'image',
        'question_key': 'question',
        'answer_key': 'answer',
    },
    "SC_NWPU": {
        'question_path': "sc/NWPU/all_questions_split.json",
        'image_path': "sc/NWPU/images_test/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "SC_SIRI": {
        'question_path': "sc/SIRI-WHU/sample_SIRI_question.json",
        'image_path': "sc/SIRI-WHU/",
        'image_key': 'image',
        'question_key': 'question',
        'answer_key': 'answer',
    },
    "SC_RS19": {
        'question_path': "sc/WHU-RS19/sample_RS19_question.json",
        'image_path': "sc/WHU-RS19/",
        'image_key': 'image',
        'question_key': 'question',
        'answer_key': 'answer',
    },
    "SC_UCM": {
        'question_path': "sc/UCMerced_split/UCMerced_split_question.json",
        'image_path': "sc/UCMerced_split/images/",
        'image_key': 'image',
        'question_key': 'question',
        'answer_key': 'answer',
    },
    # VG 4
    "VG_DIOR": {
        'question_path': "vg/DIOR-RSVG-100images/all_questions.json",
        'image_path': "vg/DIOR-RSVG-100images/images/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "VG_VRSBench": {
        'question_path': "vg/VRSBench/sample_VRSBench_all_question.json",
        'image_path': "vg/VRSBench/Images_val/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "VG_RSVG": {
        'question_path': "vg/RSVG/Q_rsvg_test_our_200.json",
        'image_path': "vg/RSVG/images/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "VG_RRSISD": {
        'question_path': "vg/RRSIS-D/Q_rrsisd_test_our_200.json",
        'image_path': "vg/RRSIS-D/JPEGImages/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    # IC
    # short
    "IC_NWPU": {
        'question_path': "ic/NWPU-Captions/sample_200_split_test_nwpu_captions.json",
        'image_path': "ic/NWPU-Captions/images/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "IC_RSICD": {
        'question_path': "ic/RSICD/sample_RSICD_images_answer_question2.json",
        'image_path': "ic/RSICD/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "IC_RSITMD": {
        'question_path': "ic/RSITMD/RSITMD_sample200_question.json",
        'image_path': "ic/RSITMD/images/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "IC_Sydney": {
        'question_path': "ic/Sydney-Captions/questions.json",
        'image_path': "ic/Sydney-Captions/images/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "IC_UCM": {
        'question_path': "ic/UCM_Captions/UCM_Captions_splite_test_question_answer.json",
        'image_path': "ic/UCM_Captions/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    # detail
    "IC_VRSBench": {
        'question_path': "ic/VRSBench/VRSBench.json",
        'image_path': "ic/VRSBench/Images_val/",
        'image_key': 'image',
        'question_key': 'question',
        'answer_key': 'answer',
    },
    # VQA
    "VQA_RSVQA-HR": {
        'question_path': "vqa/RSVQR-HR/samples_USGS_split_test_questions.json",
        'image_path': "vqa/RSVQR-HR/RSVQR-HR/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "VQA_VRSBench": {
        'question_path': "vqa/VRSBench/VRSBench_split_question.json",
        'image_path': "vqa/VRSBench/Images_val/",
        'image_key': 'image',
        'question_key': 'question',
        'answer_key': 'answer',
    },

    # OC
    "OC_HRRSD": {
        'question_path': "oc/HRRSD/sampled_300_HRRSD_all_question_answer.json",
        'image_path': "oc/HRRSD/images/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "OC_RSOD": {
        'question_path': "oc/RSOD/sample_RSOD_images_answer_question2.json",
        'image_path': "oc/RSOD/RSOD_images/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "OC_VHR": {
        'question_path': "oc/VHR/questions_vhr.json",
        'image_path': "oc/VHR/images/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "OC_VisDrone": {
        'question_path': "oc/VisionDrone/sampled_200_split_val_vision_drone.json",
        'image_path': "oc/VisionDrone/images/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },
    "OC_DOTA": {
        'question_path': "oc/DOTAv2/DOTA_Count_sample.json",
        'image_path': "oc/DOTAv2/split_images/",
        'image_key': 'image',
        'question_key': 'text',
        'answer_key': 'answer',
    },

    # Detect
    'Detect_DOTA': {
        'question_path': "detect/DOTAv2/Detect_Dota_sample.json",
        'image_path': "detect/DOTAv2/split_images/",
        'image_key': 'image',
        'question_key': 'question',
        'answer_key': 'answer',
    },
    'Detect_HRRSD': {
        'question_path': "detect/HRRSD/Detect_HRRSD_sample.json",
        'image_path': "detect/HRRSD/resize_images/",
        'image_key': 'image',
        'question_key': 'question',
        'answer_key': 'answer',
    },

}


class RSConfig(object):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        assert 'model_inference' in config and config['model_inference']
        assert 'test_dataset' in config and config['test_dataset']
        assert 'model_name' in config and config['model_name']
        assert 'save_path' in config and config['save_path']
        assert 'dataset_path' in config and config['dataset_path']
        self.model_inference = config['model_inference']
        self.test_dataset = config['test_dataset']
        self.model_name = config['model_name']
        if self.model_name == 'default':
            warnings.warn('model_name is "default"')
        self.save_path = os.path.join(config['save_path'], self.model_name)
        if os.path.exists(self.save_path):
            warnings.warn(f'save_path "{self.save_path}" already exists, files will be overwritten!')

        os.makedirs(self.save_path, exist_ok=True)
        self.log_save_path = os.path.join(config['save_path'], self.model_name, 'test_log.log')

        self.debug = False
        if 'debug' in config and config['debug']:
            self.debug = True

        self.dataset_path = config['dataset_path']
        for dataset_name, dataset_setting in all_test_dataset_dict.items():
            dataset_setting['question_path'] = os.path.join(self.dataset_path, dataset_setting['question_path'])
            dataset_setting['image_path'] = os.path.join(self.dataset_path, dataset_setting['image_path'])

        self.test_dataset_config = {}

        if 'ALL' == self.test_dataset:
            self.test_dataset_config = all_test_dataset_dict
        else:
            for dataset_name in self.test_dataset:
                task_name = dataset_name.split('_')[0]
                sub_dataset_name = dataset_name.split('_')[1]
                if sub_dataset_name == 'ALL':
                    for k_name, k_item in all_test_dataset_dict.items():
                        if task_name in k_name:
                            self.test_dataset_config[k_name] = k_item
                else:
                    assert dataset_name in all_test_dataset_dict
                    self.test_dataset_config[dataset_name] = all_test_dataset_dict[dataset_name]

        for dataset_name in self.test_dataset_config:
            task_type = dataset_name.split('_')[0]
            if task_type == 'VQA':
                self.test_dataset_config[dataset_name]['eval_func'] = VQAMetrics.VQAAccuracy
            elif task_type == 'VG':
                self.test_dataset_config[dataset_name]['eval_func'] = VGMetrics.VGIoUThreshold
            elif task_type == 'SC':
                self.test_dataset_config[dataset_name]['eval_func'] = SCMetrics.SCAccuracy
            elif task_type == 'IC':
                self.test_dataset_config[dataset_name]['eval_func'] = ICMetrics.ICAll
            elif task_type == 'OC':
                self.test_dataset_config[dataset_name]['eval_func'] = OCMetrics.OCAccuracy
            elif task_type == 'Detect':
                self.test_dataset_config[dataset_name]['eval_func'] = DetectMetrics.DetectAP
            else:
                raise RuntimeError(f'Unknown task type "{task_type}" at dataset: {dataset_name}')
        self.eval = False
        # self.metrics_setting = {}
        if 'eval' in config and config['eval']:
            assert 'metrics' in config
            self.eval = True
            self.metrics_save_path = os.path.join(config['save_path'], self.model_name, 'metrics.json')
            self.metrics_setting = config['metrics']

        if 'model_format' in config:
            self.model_format = config['model_format']


class RSLogger(object):
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, message: str):
        print(message)
        with open(self.log_path, 'a') as f:
            f.write(message + '\n')
