# We handle truth_label being 0, pred_label being 0, both being 0 case by case
def mape_2(truth_label=None, pred_label=None):
    total_loss = 0.0
    for i in range(len(truth_label)):
        if truth_label[i] != 0:
            total_loss += abs(pred_label[i] - truth_label[i]) / truth_label[i]
        elif pred_label[i] == 0:
            total_loss += 0.0
        else:
            total_loss += 1.0

    return total_loss / len(truth_label)
