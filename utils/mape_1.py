# We add 1 to the truth label to make sure the denominator is non-zero
def mape_1(truth_label=None, pred_label=None):
    total_loss = 0.0
    for i in range(len(truth_label)):
        total_loss += abs(pred_label[i] - truth_label[i]) / (truth_label[i] + 1)

    return total_loss / len(truth_label)
