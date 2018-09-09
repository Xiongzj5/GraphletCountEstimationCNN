def under_over(truth_label=None, pred_label=None):
    over_num = 0
    under_num = 0
    over_loss = 0
    under_loss = 0
    for i in range(len(truth_label)):
        if pred_label[i] > truth_label[i]:
            over_num += 1
            over_loss += abs(pred_label[i] - truth_label[i])
        elif pred_label[i] < truth_label[i]:
            under_num += 1
            under_loss += abs(pred_label[i] - truth_label[i])

    return [over_num, under_num, over_loss/(over_num+1), under_loss/(under_num+1)]
