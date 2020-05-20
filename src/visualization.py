import matplotlib.pyplot as plt  # Graph making Libraryt
from matplotlib import style
from pathlib import Path  # Path manipulation

style.use("ggplot")

model_name = "2020-05-20 13:04:05.354545"


def create_acc_loss_graph(model_name):
    model_path = Path(f"../doc/{model_name}.log")
    contents = open(model_path, "r").read().split('\n')

    times = []
    epochs = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        try:
            name, epoch, timestamp, acc, loss, val_acc, val_loss = c.split(",")
            times.append(float(timestamp))
            epochs.append(float(epoch))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))
        except Exception as e:
            pass

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1) #charts share the same axis as we zoom in for example

    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2) # loc = location

    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label="val_loss")
    ax2.legend(loc=2)  # loc = location
    plt.show()

create_acc_loss_graph(model_name)
