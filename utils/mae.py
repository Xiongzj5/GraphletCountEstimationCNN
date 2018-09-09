def mae(truth_label=None, pred_label=None):
    total_loss = 0.0
    for i in range(len(truth_label)):
        total_loss += abs(pred_label[i] - truth_label[i])

    return total_loss / len(truth_label)
