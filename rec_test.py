import torch
from tqdm import tqdm
from lib.utils.general import lev_ratio


def test(model, val_loader, criterion, conveter, device, max_i=1000):
    model.eval()

    n_correct = 0
    mloss, macc, med = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    s = ('%15s' * 4) % ('Validation:', 'Loss', 'Accuracy', 'Edit distance')

    for i, (image_batch, label) in enumerate(tqdm(val_loader, desc=s, total=max_i)):

        # Disable gradients
        with torch.no_grad():
            # Run model
            preds = model(image_batch.to(device))  # preds shape: [batch_size, 100, 6773]

            bs = image_batch.size(0)
            preds = preds.permute(1, 0, 2)
            text, length = conveter.encode(label)
            preds_lens = torch.tensor([preds.size(0)] * bs)

            loss = criterion(preds, text, preds_lens, length) / bs

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = conveter.decode(preds.data, preds_lens.data, raw=False)

        n_correct_epoch = 0
        for pred, target in zip(sim_preds, label):
            # print('pred===>', pred, 'target===>', target)
            edit_distance = lev_ratio(pred, target)
            # print('edit_distance:', edit_distance)
            if pred == target:
                n_correct += 1
                n_correct_epoch += 1
        acc = n_correct_epoch / bs

        mloss = (mloss * i + loss) / (i + 1)  # update mean losses
        macc = (macc * i + acc) / (i + 1)  # update mean acces
        med = (med * i + edit_distance) / (i + 1)  # update mean edit distance

        if i == max_i:
            break

    # Print results
    pf = '%15s' + '%15.3g' * 3  # print format
    print(pf % (' ', mloss, macc, med))

    raw_preds = conveter.decode(preds.data, preds_lens.data, raw=True)[:10]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
        print('%-20s\n%s gt:%s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_i * bs)

    return accuracy, med
