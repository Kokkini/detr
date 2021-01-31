# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import math

# %%
def extract_bbox(bbox):
    # bbox: [l, t, r, b]
    return bbox[0], bbox[1], bbox[2], bbox[3]

# %%
def scale_bbox(bbox, scale=2):
    # bbox: [l, t, r, b]
    return [i * scale for i in bbox]

# %%
def extend_bbox(img, bbox_lst, scale=2):
    # bbox_lst: array [x_topleft, y_topleft, x_botright, y_botright]
    bbox_extended = bbox_lst[:]
    bbox_extended[0] = 0
    bbox_extended[1] = bbox_extended[1] * scale
    bbox_extended[2] = img.shape[1] # 1000
    bbox_extended[3] = bbox_extended[3] * scale
    return bbox_extended

# %%
def replace_label(img, original_qa, original_label, new_label):
    # img: array
    # original_question_answer: [x_topleft, y_topleft, x_botright, y_botright]
    # original_label: [x_topleft, y_topleft, x_botright, y_botright] [0, 0, x_botright, y_botright]
    # new_label: [x_topleft, y_topleft, x_botright, y_botright]
    # Return original_image, new_image
    x_tl_0, y_tl_0, x_br_0, y_br_0 = original_label[0] - original_qa[0], original_label[1] - original_qa[1], original_label[2] - original_qa[0], original_label[3] - original_qa[1]
    x_tl_1, y_tl_1, x_br_1, y_br_1 = 0, 0, original_qa[2]- original_qa[0], original_qa[3] - original_qa[1]
    x_tl_2, y_tl_2, x_br_2, y_br_2 = new_label[0], new_label[1], new_label[2], new_label[3]

    w_0 = x_br_0 - x_tl_0
    w_1 = x_br_1 - x_tl_1
    w_2 = x_br_2 - x_tl_2
    h_0 = y_br_0 - y_tl_0
    h_1 = y_br_1 - y_tl_1
    h_2 = y_br_2 - y_tl_2

    new_img = np.array(img[original_qa[1]: original_qa[3] + 1, original_qa[0] : original_qa[2] + 1])

    # Clear the old label
    new_img[y_tl_0 : y_br_0 + 1, x_tl_0 : x_br_0 + 1] = np.ones((h_0 + 1, w_0 + 1)) * 255

    if h_2 > h_0:
        # Cut new label top pixels
        print('Cut new label top pixels')
        y_tl_2 = y_br_2 - h_0
        h_2 = h_0
        # Move all pixels down
        # new_img[y_tl_0 + h_2 + 1 : y_br_1 + 1, :] = new_img[y_br_0 + 1 : y_br_1 - (h_2 - h_0) + 1, :]
        # new_img[y_br_0 + 1 : y_tl_0 + h_2 + 1, :] = np.zeros((h_2 - h_0, w_1 + 1))
    else:
        # Ignore (white pixels already)
        pass

    if w_2 > w_0:
        # Move w_2 pixels to the right
        print('Moving pixels right')
        new_img[y_tl_0 : y_tl_0 + h_2 + 1, x_tl_0 + w_2 + 1 : ] = new_img[y_tl_0 : y_tl_0 + h_2 + 1, x_br_0 + 1 : x_br_1 - (w_2 - w_0)]

    else:
        # Ignore (white pixels already)
        pass
        
    ori_img = np.array(img[original_qa[1]: original_qa[3] + 1, original_qa[0] : original_qa[2] + 1])

    # Put the new label
    new_img[y_tl_0 : y_tl_0 + h_2 + 1, x_tl_0 : x_tl_0 + w_2 + 1] = img[y_tl_2 : y_br_2 + 1, x_tl_2 : x_br_2 + 1]
    
    return ori_img, new_img

# %%
def get_header(img, questionAns_bbox_lst):
    first_qa_bbox = round_bbox(questionAns_bbox_lst[0]["question-answer"]) # l, t, r, b
    header_l = 0
    header_t = 0
    header_r = img.shape[1]
    header_b = first_qa_bbox[1]
    return extract_image_from_bbox(img, [header_l, header_t, header_r, header_b])

# %%
def round_bbox(bbox):
    # bbox: array [left, top, right, bot]
    bbox_copy = bbox[:]
    bbox_copy[1] = math.floor(bbox_copy[1])
    bbox_copy[3] = math.floor(bbox_copy[3])
    bbox_copy[0] = math.floor(bbox_copy[0])
    bbox_copy[2] = math.floor(bbox_copy[2])
    return bbox_copy

# %%
def extract_label(questionAns_bbox_lst, scale=2):
    question_label_bbox_lst = []

    for questionAns in questionAns_bbox_lst:
        question_label_bbox = questionAns['question_label']
        question_label_bbox_extended = question_label_bbox[:]
        question_label_bbox_extended[0] = question_label_bbox_extended[0] * scale
        question_label_bbox_extended[1] = question_label_bbox_extended[1] * scale
        question_label_bbox_extended[2] = question_label_bbox_extended[2] * scale
        question_label_bbox_extended[3] = question_label_bbox_extended[3] * scale
        question_label_bbox_lst.append(question_label_bbox_extended)
    
    return question_label_bbox_lst

# %%
def replace_equal_bbox(img, bbox, new_bbox = None):
    # img: np.array
    # bbox: [l, t, r, b]
    # new_bbox: np.array same size with bbox to replace
    new_img = np.array(img)
    # bbox = scale_bbox(bbox)
    bbox = round_bbox(bbox)
    if not new_bbox: # default replace with np.255
        new_bbox = np.ones((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1)) * 255
    
    l_0, t_0, r_0, b_0 = bbox[0], bbox[1], bbox[2], bbox[3]
    new_img[t_0 : b_0 + 1, l_0 : r_0 + 1] = new_bbox
    return new_img

# %%
def replace_old_bbox(img, old_bbox, new_bbox_img, cut_top=True, cut_right=False):
    # img: np.array image
    # old_bbox: [l, t, b, r]
    # new_bbox_img: np.array image
    new_img = replace_equal_bbox(img, old_bbox)
    l_0, t_0, r_0, b_0 = extract_bbox(old_bbox)
    h_0, w_0 = b_0 - t_0 + 1, r_0 - l_0 + 1
    h_1, w_1 = new_bbox_img.shape
    
    if h_1 > h_0:
        if cut_top:
            # print('Cutting top')
            new_bbox_img = new_bbox_img[h_1 - h_0: , :]
            h_1 = h_0
    if w_1 > w_0:
        if cut_right:
            # print('Cutting right')
            new_bbox_img = new_bbox_img[:, : w_1 - (w_1 - w_0) + 1]
            w_1 = w_0
    new_img[t_0 : t_0 + h_1, l_0 : l_0 + w_1] = new_bbox_img
    return new_img

# %%
def extract_image_from_bbox(img, bbox):
    # img: np.array image
    # bbox: [l, t, r, b]
    l, t, r, b = extract_bbox(bbox)
    return np.array(img[t: b + 1, l: r + 1])

# %%
def shuffle_answers(img, question_answer):
    # img: numpy array M*N
    # question_answer: json
    answers = question_answer['answers']
    answer_Alabel = answers['A_label']
    answer_Alabel = scale_bbox(answer_Alabel)
    answer_A = answers['A']
    answer_A = scale_bbox(answer_A)
    answer_Blabel = answers['B_label']
    answer_Blabel = scale_bbox(answer_Blabel)
    answer_B = answers['B']
    answer_B = scale_bbox(answer_B)
    answer_Clabel = answers['C_label']
    answer_Clabel = scale_bbox(answer_Clabel)
    answer_C = answers['C']
    answer_C = scale_bbox(answer_C)
    answer_Dlabel = answers['D_label']
    answer_Dlabel = scale_bbox(answer_Dlabel)
    answer_D = answers['D']
    answer_D = scale_bbox(answer_D)
    # amswer = [l2 = r0 + 1, t2 = t0 = t1, r2 = r1, b2 = b0 = b1]
    answer_Abbox = [answer_Alabel[2] + 1, answer_Alabel[1], answer_A[2], answer_Alabel[3]]
    answer_Abbox = round_bbox(answer_Abbox)
    answer_Bbbox = [answer_Blabel[2] + 1, answer_Blabel[1], answer_B[2], answer_Blabel[3]]
    answer_Bbbox = round_bbox(answer_Bbbox)
    answer_Cbbox = [answer_Clabel[2] + 1, answer_Clabel[1], answer_C[2], answer_Clabel[3]]
    answer_Cbbox = round_bbox(answer_Cbbox)
    answer_Dbbox = [answer_Dlabel[2] + 1, answer_Dlabel[1], answer_D[2], answer_Dlabel[3]]
    answer_Dbbox = round_bbox(answer_Dbbox)
    answer_bbox_lst = [answer_Abbox, answer_Bbbox, answer_Cbbox, answer_Dbbox]
    answer_bbox_img_lst = []
    for i in answer_bbox_lst:
        answer_img = extract_image_from_bbox(img, i)
        answer_bbox_img_lst.append(answer_img)
    random.shuffle(answer_bbox_img_lst)

    for i, new_answer in enumerate(answer_bbox_img_lst):
        old_answer = answer_bbox_lst[i]
        # old_answer = scale_bbox(old_answer)
        old_answer = round_bbox(old_answer)
        # Remove old_answer
        # img = replace_equal_bbox(img, old_answer)
        # Replace with new_answer
        img = replace_old_bbox(img, old_answer, new_answer)
    return img

# %%
def fill_footer(new_img, ori_img_height, ori_img_width):
    new_img_height, new_img_width = new_img.shape
    # print('new img height:', new_img_height)
    # print('ori img height:', ori_img_height)
    filler = np.ones((ori_img_height - new_img_height, new_img_width)) * 255
    new_img = np.concatenate((new_img, filler))
    return new_img

# %%
class Mixer:
    def __init__(self):
        return
    
    def mix(self, img, bounding_box_dct, scale=2):
        """mix the questions and the answers for each question

        Args:
            img (ndarray): 3-channel image
            anno (dict): annotaiton

        Returns:
            ndarray: mixed image
        """
        img = np.array(img)
        ori_img_height, ori_img_width = img.shape[:2]
        questionAns_bbox_lst = bounding_box_dct['question-answers']

        #
        # Calculate distance between each question (a space / y_topleft - y_bottomright)
        dist_between_boxes = 0
        for i in range(1, len(questionAns_bbox_lst)):
            curr_questionAn_bbox = questionAns_bbox_lst[i]['question-answer']
            prev_questionAn_bbox = questionAns_bbox_lst[i - 1]['question-answer']
            curr_y_top_left = curr_questionAn_bbox[1] * scale
            prev_y_bottom_right = prev_questionAn_bbox[3] * scale
            dist_between_boxes += (curr_y_top_left - prev_y_bottom_right)
        dist_between_boxes = int(dist_between_boxes / (len(questionAns_bbox_lst) - 1))

        #
        # # Extend bounding box for each question
        # questionAns_bbox_extended_lst = []

        # for questionAns in questionAns_bbox_lst:
        #     questionAn_bbox = questionAns['question-answer']
        #     questionAn_bbox_extended = questionAn_bbox[:]
        #     questionAn_bbox_extended[0] = 0
        #     questionAn_bbox_extended[1] = questionAn_bbox_extended[1] * scale
        #     questionAn_bbox_extended[2] = img.shape[1] # 1000
        #     questionAn_bbox_extended[3] = questionAn_bbox_extended[3] * scale
        #     questionAns_bbox_extended_lst.append(questionAn_bbox_extended)

        #
        # Extract 'Question Label'
        question_label_bbox_lst = []

        question_label_bbox_lst = extract_label(questionAns_bbox_lst)

        #
        # Rescale question_label based on median
        ''' This shouldn't work because what of question 1: and question 10: would have different length and adjusting to one length would either cut another character or add another character'''
        ''' Better way should be ensuring that the bounding boxes are correct by OCR'''
        # question_label_width_bbox_lst = []
        # question_label_height_bbox_lst = []

        # for question_label in question_label_bbox_lst:
        #     question_label_width_bbox_lst.append(question_label[2] - question_label[0])
        #     question_label_height_bbox_lst.append(question_label[3] - question_label[1])

        # question_label_median_width = np.median(np.array(question_label_width_bbox_lst))
        # question_label_median_height = np.median(np.array(question_label_height_bbox_lst))

        #
        # For loop => new img
        new_img = np.ones((1, img.shape[1])) * 255
        dist_box = np.ones((dist_between_boxes, img.shape[1])) * 255


        # Check for header
        header = None
        if 'header' in bounding_box_dct:
            header = bounding_box_dct['header'][:]
            # Extend header
            header[0] = 0
            header[1] = header[1] * scale
            header[2] = img.shape[1]
            header[3] = header[3] * scale

        # Add header
        if header is not None:
            row_start = 0
            row_end = math.ceil(header[3])  + 1
            col_start = 0
            col_end = math.ceil(header[2]) + 1
            new_img = img[row_start: row_end, col_start: col_end]
            new_img = np.concatenate((new_img, dist_box))
        else:
            header = get_header(img, questionAns_bbox_lst)
            new_img = header
            # savetest = 'header.jpg'
            # cv2.imwrite(savetest, header)

        random.shuffle(questionAns_bbox_lst)

        # Concatenate
        count = 0
        for i, questionAns in enumerate(questionAns_bbox_lst):
            img = shuffle_answers(img, questionAns)
            # savetest = './' + str(count) + '.jpg'
            # cv2.imwrite('./test2.jpg', img)
            # cv2.imwrite(savetest, img)
            count += 1
            question_answer_bbox = questionAns['question-answer']
            question_answer_bbox_extended = extend_bbox(img, question_answer_bbox)
            question_label_bbox = questionAns['question_label'][:]
            question_label_bbox = [i * scale for i in question_label_bbox]
            new_label_bbox = question_label_bbox_lst[i]

            # row_start = math.floor(question_answer_bbox_extended[1])
            # row_end = math.ceil(question_answer_bbox_extended[3])  + 1
            # col_start = math.floor(question_answer_bbox_extended[0])
            # col_end = math.ceil(question_answer_bbox_extended[2]) + 1

            question_answer_bbox_extended = round_bbox(question_answer_bbox_extended)
            question_label_bbox = round_bbox(question_label_bbox)
            new_label_bbox = round_bbox(new_label_bbox)
            
            # new_img[row_start: row_end, col_start: col_end] = img[row_start: row_end, col_start: col_end]
            # question_img = img[row_start: row_end, col_start: col_end]
            ori_question_img, new_question_img = replace_label(img, question_answer_bbox_extended, question_label_bbox, new_label_bbox)

            new_img = np.concatenate((new_img, new_question_img))
            # new_img = np.concatenate((new_img, question_img))
            if i < len(questionAns_bbox_lst) - 1:
                new_img = np.concatenate((new_img, dist_box))

        new_img = fill_footer(new_img, ori_img_height, ori_img_width)
        return new_img