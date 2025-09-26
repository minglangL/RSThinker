import re
import numpy as np


class VQAMetrics:
    @staticmethod
    def AccuracySingle(answer: str, target: str):
        count_transfer = {'0': 'no', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six',
                          '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten', '11': 'eleven', '12': 'twelve'}

        answer = answer.lower()
        target = target.lower()

        return target in answer or (count_transfer.get(target) and count_transfer[target] in answer)

    @staticmethod
    def VQAAccuracy(answer_list: [str], target_list: [str], **kwargs):
        assert len(answer_list) == len(target_list)
        try:
            acc_count = 0
            for i in range(len(answer_list)):
                answer = answer_list[i]
                target = target_list[i]
                if VQAMetrics.AccuracySingle(answer, target):
                    acc_count += 1

            acc = acc_count / len(answer_list)
        except Exception as e:
            print(e)
            acc = 0
        return {'acc': acc}


class VGMetrics:
    @staticmethod
    def calculate_iou(box1, box2, transfer: bool = True):
        # box格式: [xmin, ymin, xmax, ymax]

        if len(box1) < 4 or len(box2) < 4:
            return 0

        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        if transfer:
            x2, y2 = x1 + x2, y1 + y2
            x2_p, y2_p = x1_p + x2_p, y1_p + y2_p

        x1_inter = max(x1, x1_p)
        y1_inter = max(y1, y1_p)
        x2_inter = min(x2, x2_p)
        y2_inter = min(y2, y2_p)
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area
        return iou

    @staticmethod
    def extract_all_floats(input_string):
        floats = re.findall(r'-?\d*\.\d+|-?\d+', input_string)
        floats = [float(item) for item in floats]
        if len(floats) >= 4:
            return floats[:4]
        else:
            return []

    @staticmethod
    def VGIoUThreshold(answer_list: [str], target_list: [str], answer_type: str = 'pix', image_size=(800, 800),
                       iou_threshold_list=None):
        try:
            if iou_threshold_list is None:
                iou_threshold_list = [0.5, 0.75, 0.95]
            assert len(answer_list) == len(target_list)
            width, height = image_size

            def percentage_transfer(answer_box):
                radio = 100
                return [answer_box[0] / radio * width, answer_box[1] / radio * height, answer_box[2] / radio * width,
                        answer_box[3] / radio * height]

            def per_thousand_transfer(answer_box):
                radio = 1000
                return [answer_box[0] / radio * width, answer_box[1] / radio * height, answer_box[2] / radio * width,
                        answer_box[3] / radio * height]

            answer_transfer = None
            if answer_type == 'percentage':
                answer_transfer = percentage_transfer
            elif answer_type == 'per_thousand':
                answer_transfer = per_thousand_transfer

            acc_dict = {}
            for iou_threshold in iou_threshold_list:
                acc_dict[iou_threshold] = 0

            iou_list = []

            for i in range(len(answer_list)):
                answer = answer_list[i]
                target = target_list[i]

                answer = VGMetrics.extract_all_floats(answer)
                if answer_transfer and len(answer) >= 4:
                    answer = answer_transfer(answer)
                target = VGMetrics.extract_all_floats(target)
                if answer and target:
                    iou = VGMetrics.calculate_iou(answer, target)
                else:
                    iou = 0.0

                iou_list.append(iou)
                for iou_threshold in iou_threshold_list:
                    if iou >= iou_threshold:
                        acc_dict[iou_threshold] += 1

            for iou_threshold in acc_dict.keys():
                acc_dict[iou_threshold] = acc_dict[iou_threshold] / len(answer_list)

            mean_iou = sum(iou_list) / len(iou_list)
            acc_dict['mean_iou'] = mean_iou
        except Exception as e:
            print(e)
            acc_dict = {}
        return acc_dict


class SCMetrics:
    @staticmethod
    def AccuracySingle(answer: str, target: str):
        answer = answer.lower()
        target = target.lower()

        answer = answer.replace('_', '').replace('-', '').replace(' ','')
        target = target.replace('_', '').replace('-', '').replace(' ','')

        return target in answer

    @staticmethod
    def SCAccuracy(answer_list: [str], target_list: [str]):
        assert len(answer_list) == len(target_list)
        try:
            acc_count = 0
            for i in range(len(answer_list)):
                answer = answer_list[i]
                target = target_list[i]
                if SCMetrics.AccuracySingle(answer, target):
                    acc_count += 1

            acc = acc_count / len(answer_list)
        except Exception as e:
            print(e)
            acc = 0
        return {'acc': acc}


class OCMetrics:
    @staticmethod
    def extract_number(answer_str: str):
        def extract_first_number_sequences(input_string):
            numbers = re.findall(r'\d+', input_string)  # 找到所有数字序列
            if len(numbers) >= 1:
                return int(numbers[0])
            else:
                return -1

        numbers_list = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

        answer_str = answer_str.lower()

        answer = extract_first_number_sequences(answer_str)
        if answer == -1:
            for j in range(len(numbers_list)):
                if numbers_list[j] in answer_str:
                    answer = j
                    break

        return answer

    @staticmethod
    def OCAccuracy(answer_list: [str], target_list: [str], max_min_list=None):
        try:
            if max_min_list is None:
                max_min_list = [0.0, 0.1, 0.3, 0.5]
            assert len(answer_list) == len(target_list)

            acc_dict = {}
            for max_min in max_min_list:
                acc_dict[max_min] = 0
            mae_list = []

            for i in range(len(answer_list)):
                answer = answer_list[i]
                target = target_list[i]
                answer = OCMetrics.extract_number(answer)
                target = int(target)
                mae = abs(target - answer)

                for max_min in max_min_list:
                    if mae <= max_min * target:
                        acc_dict[max_min] += 1
                mae_list.append(mae)

            for max_min in acc_dict.keys():
                acc_dict[max_min] = acc_dict[max_min] / len(answer_list)

            mean_mae = sum(mae_list) / len(answer_list)
            acc_dict['mean_mae'] = mean_mae
        except Exception as e:
            print(e)
            print(answer_list, target_list, max_min_list)
            acc_dict = {}
        return acc_dict


class ICMetrics:

    @staticmethod
    def compute_rouge_l(reference, candidate):
        from rouge_score import rouge_scorer
        scorer_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        reference = reference[0]
        scores = scorer_rouge.score(reference, candidate)
        # print(scores)
        return scores['rougeL'].fmeasure

    @staticmethod
    def compute_cider(reference, candidate):
        import numpy as np
        from collections import Counter
        from nltk.util import ngrams
        from sklearn.feature_extraction.text import TfidfVectorizer

        def ngram_similarity(candidate, reference, n):
            candidate_ngrams = list(ngrams(candidate.split(), n))
            reference_ngrams = list(ngrams(reference.split(), n))
            candidate_counter = Counter(candidate_ngrams)
            reference_counter = Counter(reference_ngrams)
            common = candidate_counter & reference_counter
            return sum(common.values()) / max(len(candidate_ngrams), len(reference_ngrams))

        def tfidf_weighted_similarity(candidate, reference, n):
            vectorizer = TfidfVectorizer(ngram_range=(n, n))
            tfidf_matrix = vectorizer.fit_transform([candidate, reference])
            return np.dot(tfidf_matrix[0].toarray(), tfidf_matrix[1].toarray().T)[0][0]
        try:
            reference = reference[0]

            similarity = 0
            for n in range(1, 5):  # 计算1-gram到4-gram的相似度
                similarity += ngram_similarity(candidate, reference, n)
                similarity += tfidf_weighted_similarity(candidate, reference, n)
            similarity /= 8  # 平均化
        except Exception as e:
            print(e)
            similarity = 0
        # print(similarity)
        return similarity

    @staticmethod
    def compute_meteor(reference, candidate):
        from nltk.translate import meteor_score
        try:
            reference_texts = [it.split() for it in reference]
            generated_text = candidate.split()
            meteor = meteor_score.meteor_score(reference_texts, generated_text)
        except Exception as e:
            print(e)
            meteor = 0
        return meteor

    @staticmethod
    def individual_bleu(reference, candidate):
        from nltk.translate.bleu_score import sentence_bleu
        try:
            bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            bleu_2_gram = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
            bleu_3_gram = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
            bleu_4_gram = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
        except Exception as e:
            print(e)
            bleu_1_gram = 0
            bleu_2_gram = 0
            bleu_3_gram = 0
            bleu_4_gram = 0

        # print('bleu 1-gram: %f' % bleu_1_gram)
        # print('bleu 2-gram: %f' % bleu_2_gram)
        # print('bleu 3-gram: %f' % bleu_3_gram)
        # print('bleu 4-gram: %f' % bleu_4_gram)

        return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram

    @staticmethod
    def cumulative_bleu(reference, candidate):
        from nltk.translate.bleu_score import sentence_bleu
        try:
            bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            bleu_2_gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
            bleu_3_gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
            bleu_4_gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        except Exception as e:
            print(e)
            bleu_1_gram = 0
            bleu_2_gram = 0
            bleu_3_gram = 0
            bleu_4_gram = 0

        # print('bleu 1-gram: %f' % bleu_1_gram)
        # print('bleu 2-gram: %f' % bleu_2_gram)
        # print('bleu 3-gram: %f' % bleu_3_gram)
        # print('bleu 4-gram: %f' % bleu_4_gram)

        return bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram

    @staticmethod
    def ICAll(answer_list: [str], target_list: [str]):
        assert len(answer_list) == len(target_list)
        try:
            result = {
                'bleu1': [],
                'bleu2': [],
                'bleu3': [],
                'bleu4': [],
                'meteor': [],
                'cider': [],
                'rouge_l': [],
            }

            for i in range(len(answer_list)):
                answer = answer_list[i]
                target = target_list[i]
                if type(target) is str:
                    target = [target]

                answer = re.sub(r'\{[^}]*\}|\<[^\>]*\>', '', answer)

                bleu1, bleu2, bleu3, bleu4 = ICMetrics.cumulative_bleu(target, answer)
                meteor = ICMetrics.compute_meteor(target, answer)
                cider = ICMetrics.compute_cider(target, answer)
                rouge = ICMetrics.compute_rouge_l(target, answer)

                result['bleu1'].append(bleu1)
                result['bleu2'].append(bleu2)
                result['bleu3'].append(bleu3)
                result['bleu4'].append(bleu4)
                result['meteor'].append(meteor)
                result['cider'].append(cider)
                result['rouge_l'].append(rouge)

                # print(f'answer:{answer}, target:{target}')
                # print(f'bleu1-4:{bleu1},{bleu2},{bleu3},{bleu4},meteor:{meteor},cider:{cider},rouge_l:{rouge}')

            metrics_dict = {}
            for metric in result.keys():
                metrics_dict[metric] = sum(result[metric]) / len(result[metric])
        except Exception as e:
            print(e)
            metrics_dict = {}
        return metrics_dict


class DetectMetrics:
    @staticmethod
    def extract_all_boxes(input_string):
        floats = re.findall(r'-?\d*\.\d+|-?\d+', input_string)
        floats = [float(item) for item in floats]
        box_list = []
        for i in range(0, len(floats), 4):
            box_list.append([floats[i], floats[i + 1], floats[i + 2], floats[i + 3]])
        return box_list

    @staticmethod
    def calculate_ap(detected_boxes, ground_truth_boxes, iou_threshold=0.5):
        true_positives = np.zeros(len(detected_boxes))
        false_positives = np.zeros(len(detected_boxes))

        # 用于记录真实框是否已被匹配
        ground_truth_matched = [False] * len(ground_truth_boxes)

        for i, det_box in enumerate(detected_boxes):
            best_iou = 0
            best_gt_index = -1

            for j, gt_box in enumerate(ground_truth_boxes):
                iou = VGMetrics.calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = j

            if best_iou > iou_threshold and not ground_truth_matched[best_gt_index]:
                true_positives[i] = 1
                ground_truth_matched[best_gt_index] = True
            else:
                false_positives[i] = 1

        # 计算累积TP和FP
        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)

        # 计算召回率和精确率
        recall = cumulative_tp / len(ground_truth_boxes)
        precision = cumulative_tp / (cumulative_tp + cumulative_fp)

        # 计算AP
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11

        return ap

    @staticmethod
    def f1_score(pred_boxes, gt_boxes, iou_threshold=0.5):
        matched_gt = set()
        TP = 0
        for pb in pred_boxes:
            best_iou = 0
            best_gt = -1
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou_val = VGMetrics.calculate_iou(pb, gb)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = i
            if best_iou >= iou_threshold:
                TP += 1
                matched_gt.add(best_gt)

        FP = len(pred_boxes) - TP
        FN = len(gt_boxes) - TP

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1

    @staticmethod
    def DetectAP(answer_list: [str], target_list, answer_type: str = 'pix', image_size=(800, 800),
                 iou_threshold_list=None, answer_box_extract_fuc=None):
        try:
            if iou_threshold_list is None:
                iou_threshold_list = [0.25, 0.5, 0.75]
            assert len(answer_list) == len(target_list)
            width, height = image_size

            def percentage_transfer(answer_box):
                radio = 100
                return [answer_box[0] / radio * width, answer_box[1] / radio * height, answer_box[2] / radio * width,
                        answer_box[3] / radio * height]

            def per_thousand_transfer(answer_box):
                radio = 1000
                return [answer_box[0] / radio * width, answer_box[1] / radio * height, answer_box[2] / radio * width,
                        answer_box[3] / radio * height]

            answer_transfer = None
            if answer_type == 'percentage':
                answer_transfer = percentage_transfer
            elif answer_type == 'per_thousand':
                answer_transfer = per_thousand_transfer

            ap_dict = {}
            f1_dict = {}
            for iou_threshold in iou_threshold_list:
                ap_dict[iou_threshold] = []
                f1_dict[iou_threshold] = {}
                f1_dict[iou_threshold]['precision'] = []
                f1_dict[iou_threshold]['recall'] = []
                f1_dict[iou_threshold]['f1'] = []

            for i in range(len(answer_list)):
                answer = answer_list[i]
                target = target_list[i]

                answer_box_list = DetectMetrics.extract_all_boxes(answer)
                if answer_transfer:
                    answer_box_list = [answer_transfer(box) for box in answer_box_list]

                target_box_list = target

                # print(f'target: {target_box_list}, answer_box_list: {answer_box_list}')

                for iou_threshold in iou_threshold_list:
                    ap = DetectMetrics.calculate_ap(answer_box_list, target_box_list, iou_threshold)
                    ap_dict[iou_threshold].append(ap)
                    precision, recall, f1 = DetectMetrics.f1_score(answer_box_list, target_box_list, iou_threshold)
                    f1_dict[iou_threshold]['precision'].append(precision)
                    f1_dict[iou_threshold]['recall'].append(recall)
                    f1_dict[iou_threshold]['f1'].append(f1)

            metrics_dict = {}
            for iou_threshold in ap_dict.keys():
                metrics_dict[iou_threshold] = {
                    'ap': sum(ap_dict[iou_threshold]) / len(ap_dict[iou_threshold]),
                    'f1': sum(f1_dict[iou_threshold]['f1']) / len(f1_dict[iou_threshold]['f1']),
                    'recall': sum(f1_dict[iou_threshold]['recall']) / len(f1_dict[iou_threshold]['recall']),
                    'precision': sum(f1_dict[iou_threshold]['precision']) / len(f1_dict[iou_threshold]['precision']),
                }
        except Exception as e:
            print(e)
            metrics_dict = {}
        return metrics_dict

