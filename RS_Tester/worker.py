import json
import os
import yaml
from tqdm import tqdm
from functools import partial
from RS_Tester.config import RSConfig, RSLogger


class WorkerFlower(object):
    def __init__(self, config_path, model_inference, model_name=None):
        assert os.path.exists(config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['model_inference'] = model_inference
        if model_name is not None:
            config['model_name'] = model_name

        assert config['model_name'] is not None, 'please specify a model name'
        self.config = RSConfig(config)

        self.loggerClass = RSLogger(self.config.log_save_path)
        self.logger = self.loggerClass.log

    def run(self):
        test_dataset = self.config.test_dataset_config
        model_infer = self.config.model_inference
        self.logger(f'All Test Dataset:\n {test_dataset.keys()}')
        for dataset_name, dataset_setting in test_dataset.items():
            self.logger(f'Testing dataset {dataset_name}')
            question_json = []
            with open(dataset_setting['question_path'], "r", encoding='utf-8') as f:
                for line in f.readlines():
                    question_json.append(json.loads(line))

            dataset_image_path = dataset_setting['image_path']
            image_key = dataset_setting['image_key']
            question_key = dataset_setting['question_key']
            answers_key = dataset_setting.get('answer_key', None)

            out_json = []
            for question_item in tqdm(question_json):
                image_path = os.path.join(dataset_image_path, question_item[image_key])
                assert os.path.exists(image_path), f'{image_path} does not exist'
                model_answer = model_infer(image_path, question_item[question_key])
                if self.config.debug:
                    print(model_answer)

                out_item = question_item.copy()
                out_item['model_answer'] = model_answer
                out_json.append(out_item)

            json_save_path = os.path.join(self.config.save_path, f'{dataset_name}_{self.config.model_name}.json')
            self.logger(f'Completed Testing Dataset: {dataset_name}, Save To:{json_save_path}')
            with open(json_save_path, 'w') as f:
                for item in out_json:
                    f.write(json.dumps(item) + '\n')

            if self.config.eval:
                self.logger(f'Compute Metrics')
                if answers_key:
                    task_type = dataset_name.split('_')[0]

                    if type(out_json[0]['model_answer']) == str:
                        answer_list = [item['model_answer'] for item in out_json]
                    else:
                        answer_list = [item['model_answer']['answer'] for item in out_json]

                    target_list = [item[answers_key] for item in out_json]
                    eval_func = dataset_setting['eval_func']
                    extra_kwargs = self.config.metrics_setting.get(task_type, {})
                    eval_metric = eval_func(answer_list, target_list, **extra_kwargs)
                    self.logger(f'Eval Metrics: {eval_metric}')
                else:
                    self.logger(f'Dataset is not supported, answers_key({answers_key}) is None.')
