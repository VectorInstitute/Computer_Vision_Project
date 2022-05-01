import os
import torch
import time
from core.utils import *
from dataset_factory.meters import AVAMeter
import wandb
from tqdm import tqdm


def train_ava(cfg, epoch, model, train_loader, loss_module, optimizer):
    t0 = time.time()
    loss_module.reset_meters()
    l_loader = len(train_loader)

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data = batch['clip'].cuda()
        target = {'cls': batch['cls'], 'boxes': batch['boxes']}
        output = model(data)
        loss = loss_module(output, target, epoch, batch_idx, l_loader)

        loss.backward()
        steps = cfg.TRAIN.TOTAL_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
        if batch_idx % steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # save result every 1000 batches
        if batch_idx % 2000 == 0: # From time to time, reset averagemeters to see improvements
            loss_module.reset_meters()


    t1 = time.time()
    logging('trained with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    print('')
   


def train(cfg, epoch, model, train_loader, loss_module, optimizer,
                        batch=0, seed=None, score=None, is_best=None, checkpointer=None):
    t0 = time.time()
    loss_module.reset_meters()
    l_loader = len(train_loader)

    model.train()
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, obj_target, anomaly_target) in enumerate(tqdm(train_loader)):
        if batch_idx < batch:
            continue
        
        data = data.cuda()
        anomaly_target = anomaly_target.cuda()
        output, anomaly_logits = model(data)

        # bbox loss
        loss_bbox = loss_module(output, obj_target, epoch, batch_idx, l_loader)

        # CE loss (anomaly)
        loss_ce = cross_entropy(anomaly_logits, anomaly_target)


        loss = loss_bbox  + cfg.SOLVER.CE_LOSS_WEIGHT * loss_ce

        loss.backward()
        steps = cfg.TRAIN.TOTAL_BATCH_SIZE // cfg.TRAIN.BATCH_SIZE
        if batch_idx % steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        wandb.log({'global_step': epoch*l_loader, 'bbox_loss': loss_bbox, 'ce_loss': loss_ce, 'total_loss': loss})
        # save result every 1000 batches
        if batch_idx % 2000 == 0: # From time to time, reset averagemeters to see improvements
            loss_module.reset_meters()
        
    t1 = time.time()
    logging('trained with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    print('')



@torch.no_grad()
def test_ava(cfg, epoch, model, test_loader):
     # Test parameters
    num_classes       = cfg.MODEL.NUM_CLASSES
    anchors           = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors       = cfg.SOLVER.NUM_ANCHORS
    nms_thresh        = 0.5
    conf_thresh_valid = 0.005

    nbatch = len(test_loader)
    meter = AVAMeter(cfg, cfg.TRAIN.MODE, 'latest_detection.json')

    model.eval()
    for batch_idx, batch in enumerate(test_loader):
        data = batch['clip'].cuda()
        target = {'cls': batch['cls'], 'boxes': batch['boxes']}

        with torch.no_grad():
            output = model(data)
            metadata = batch['metadata'].cpu().numpy()

            preds = []
            all_boxes = get_region_boxes_ava(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)
                
                for box in boxes:
                    x1 = float(box[0]-box[2]/2.0)
                    y1 = float(box[1]-box[3]/2.0)
                    x2 = float(box[0]+box[2]/2.0)
                    y2 = float(box[1]+box[3]/2.0)
                    det_conf = float(box[4])
                    cls_out = [det_conf * x.cpu().numpy() for x in box[5]]
                    preds.append([[x1,y1,x2,y2], cls_out, metadata[i][:2].tolist()])

        meter.update_stats(preds)
        logging("[%d/%d]" % (batch_idx, nbatch))

    mAP = meter.evaluate_ava()
    logging("mode: {} -- mAP: {}".format(meter.mode, mAP))

    return mAP

@torch.no_grad()
def test_ucf24_jhmdb21(cfg, epoch, model, test_loader):

    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Test parameters
    nms_thresh    = 0.4
    iou_thresh    = 0.5
    eps           = 1e-5
    num_classes = cfg.MODEL.NUM_CLASSES
    anchors     = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors = cfg.SOLVER.NUM_ANCHORS
    conf_thresh_valid = 0.005
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    fscore = 0.0

    correct_classification = 0.0
    total_detected = 0.0

    nbatch = len(test_loader)

    model.eval()

    for batch_idx, (frame_idx, data, obj_target, anomaly_target) in enumerate(test_loader):
        data = data.cuda()
        with torch.no_grad():
            output = model(data).data
            all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)
                if cfg.TRAIN.DATASET == 'ucf24':
                    detection_path = os.path.join('ucf_detections', 'detections_'+str(epoch), frame_idx[i])
                    current_dir = os.path.join('ucf_detections', 'detections_'+str(epoch))
                    if not os.path.exists('ucf_detections'):
                        os.mkdir('ucf_detections')
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)
                else:
                    detection_path = os.path.join('jhmdb_detections', 'detections_'+str(epoch), frame_idx[i])
                    current_dir = os.path.join('jhmdb_detections', 'detections_'+str(epoch))
                    if not os.path.exists('jhmdb_detections'):
                        os.mkdir('jhmdb_detections')
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)

                with open(detection_path, 'w+') as f_detect:
                    for box in boxes:
                        x1 = round(float(box[0]-box[2]/2.0) * 320.0)
                        y1 = round(float(box[1]-box[3]/2.0) * 240.0)
                        x2 = round(float(box[0]+box[2]/2.0) * 320.0)
                        y2 = round(float(box[1]+box[3]/2.0) * 240.0)

                        det_conf = float(box[4])
                        for j in range((len(box)-5)//2):
                            cls_conf = float(box[5+2*j].item())
                            prob = det_conf * cls_conf

                            f_detect.write(str(int(box[6])+1) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
                truths = obj_target[i].view(-1, 5)
                num_gts = truths_length(truths)
        
                total = total + num_gts
            
                # ADD BOXES WITH OVER 0.25 CONF TO pred_list TO BE USED DURING 
                pred_list = [] # LIST OF CONFIDENT BOX INDICES
    
                for i in range(len(boxes)):
                    if boxes[i][4] > 0.25:
                        proposals = proposals+1
                        pred_list.append(i)

                for i in range(num_gts):
                    box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                    best_iou = 0
                    best_j = -1
                    #for j in range(len(boxes)):
                    for j in pred_list: # ITERATE THROUGH ONLY CONFIDENT BOXES
                        iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou

                    if best_iou > iou_thresh:
                        total_detected += 1
                        if int(boxes[best_j][6]) == box_gt[6]:
                            correct_classification += 1

                    if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
                        correct = correct+1

            precision = 1.0*correct/(proposals+eps)
            recall = 1.0*correct/(total+eps)
            fscore = 2.0*precision*recall/(precision+recall+eps)
            logging("[%d/%d] precision: %f, recall: %f, fscore: %f" % (batch_idx, nbatch, precision, recall, fscore))
            
    classification_accuracy = 1.0 * correct_classification / (total_detected + eps)
    locolization_recall = 1.0 * total_detected / (total + eps)

    print("Classification accuracy: %.3f" % classification_accuracy)
    print("Locolization recall: %.3f" % locolization_recall)

    return fscore

@torch.no_grad()
def test_dota(cfg, epoch, model, test_loader):

    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Test parameters
    nms_thresh    = 0.4
    iou_thresh    = 0.5
    eps           = 1e-5
    num_classes   = cfg.MODEL.NUM_CLASSES if cfg.SOLVER.COMBINED_BOX else cfg.LISTDATA.MAX_OBJS
    anchors     = [float(i) for i in cfg.SOLVER.ANCHORS]
    num_anchors = cfg.SOLVER.NUM_ANCHORS
    conf_thresh_valid = 0.005
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    fscore = 0.0
    box_conf_thresh = 0.25

    correct_classification = 0.0
    total_detected = 0.0

    anomaly_classification = 0.0
    num_anomalies = 0.0
    nbatch = len(test_loader)

    model.eval()


    per_class_tot = {'1': 0.0, '2': 0.0, '3': 0.0,
                      '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0,
                      '8': 0.0, '9': 0.0, '10': 0.0, '11': 0.0}
    per_class_pred = {'1': 0.0, '2': 0.0, '3': 0.0,
                      '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0,
                      '8': 0.0, '9': 0.0, '10': 0.0, '11': 0.0}


    for batch_idx, (frame_idx, data, target, anomaly_cls_target) in enumerate(test_loader):
        data = data.cuda()
        with torch.no_grad():
            output, anomaly_output = model(data)

            # # bbox loss
            # loss_bbox = loss_module(output, obj_target, epoch, batch_idx, l_loader)
            #
            # # CE loss (anomaly)
            # loss_ce = cross_entropy(anomaly_logits, anomaly_target)


            output = output.data
            all_boxes = get_region_boxes(output, conf_thresh_valid, num_classes, anchors, num_anchors, 0, 1)
            for i in range(output.size(0)):
                num_anomalies += 1
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)

                if not os.path.exists('dota_detections'):
                    os.mkdir('dota_detections')

                detection_dir = 'dota_detections/' + cfg.BACKUP_DIR.split('/')[-1]

                if not os.path.exists(detection_dir):
                    os.mkdir(detection_dir)

                detection_path = detection_dir + '/detections_'+str(epoch)

                if not os.path.exists(detection_path):
                    os.mkdir(detection_path)

                detection_path = detection_path + '/' + frame_idx[i]


                with open(detection_path, 'w+') as f_detect:
                    for box in boxes:
                        x1 = round(float(box[0]-box[2]/2.0) * 224.0)
                        y1 = round(float(box[1]-box[3]/2.0) * 224.0)
                        x2 = round(float(box[0]+box[2]/2.0) * 224.0)
                        y2 = round(float(box[1]+box[3]/2.0) * 224.0)

                        det_conf = float(box[4])
                        for j in range((len(box)-5)//2):
                            cls_conf = float(box[5+2*j].item())
                            prob = det_conf * cls_conf

                            f_detect.write(str(int(box[6])+1) + ' ' + str(prob) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2)  + ' ' +  str(int(anomaly_cls_target[i]) +1) + '\n')
                truths = target[i].view(-1, 5)
                num_gts = truths_length(truths)
        
                total = total + num_gts
            
                # ADD BOXES WITH OVER 0.25 CONF TO pred_list TO BE USED DURING 
                pred_list = [] # LIST OF CONFIDENT BOX INDICES
                pred_boxes = []
                for j in range(len(boxes)):
                    if boxes[j][4] > box_conf_thresh:
                        proposals = proposals+1
                        pred_list.append(j)
                        pred_boxes.append(boxes[j])

                for k in range(num_gts):
                    box_gt = [truths[k][1], truths[k][2], truths[k][3], truths[k][4], 1.0, 1.0, truths[k][0]]
                    best_iou = 0
                    best_j = -1
                    #for j in range(len(boxes)):
                    for j in pred_list: # ITERATE THROUGH ONLY CONFIDENT BOXES
                        iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou

                    if best_iou > iou_thresh:
                        total_detected += 1
                        if int(boxes[best_j][6]) == box_gt[6]:
                            correct_classification += 1

                    if best_iou > iou_thresh and int(boxes[best_j][6]) == box_gt[6]:
                        correct = correct+1

                anom_cls = str(anomaly_cls_target[i].item())
                per_class_tot[anom_cls] += 1

                if torch.argmax(anomaly_output, dim=1)[i].item() == anomaly_cls_target[i].item():
                    anomaly_classification += 1
                    anom_pred = str(torch.argmax(anomaly_output, dim=1)[i].item())
                    per_class_pred[anom_cls] += 1

            precision = 1.0*correct/(proposals+eps)
            recall = 1.0*correct/(total+eps)
            fscore = 2.0*precision*recall/(precision+recall+eps)

            print("[%d/%d] precision: %f, recall: %f, fscore: %f" % (batch_idx, nbatch, precision, recall, fscore))

            
    classification_accuracy = 1.0 * correct_classification / (total_detected + eps)
    localization_recall = 1.0 * total_detected / (total + eps)
    anomaly_classification_acc = 1.0 * anomaly_classification / num_anomalies

    wandb.log({'object_precision': precision})
    wandb.log({'object_recall': recall})
    wandb.log({'object_classification_accuracy': classification_accuracy})
    wandb.log({'object_localization_recall': localization_recall})
    wandb.log({'anomaly_classification_accuracy': anomaly_classification_acc})

    per_class_acc = {}

    for cls in per_class_tot:
        name = 'class_' + cls + 'acc'
        if per_class_tot[cls] > 0:
            per_class_acc[name] = per_class_pred[cls] / per_class_tot[cls]
        else:
            per_class_acc[name] = 0.0

    wandb.log(per_class_acc)

    print("Classification accuracy: %.3f" % classification_accuracy)
    print("Locolization recall: %.3f" % localization_recall)
    print("Anomaly Classification Accuracy: %.3f" % anomaly_classification_acc)

    return fscore


