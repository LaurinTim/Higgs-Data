import matplotlib.pyplot as plt
import numpy as np

def plot_training_info(train_loss, valid_loss, train_auc, valid_auc, n=300, skip_iterations=0) -> None:
    total_epochs = len(valid_loss)
    
    train_loss_truncated = np.array(train_loss[:(len(train_loss) - (len(train_loss) % n))]).reshape(-1, n).mean(axis=1)
    train_auc_truncated = np.array(train_auc[:(len(train_auc) - (len(train_auc) % n))]).reshape(-1, n).mean(axis=1)

    x_train = np.linspace(0, total_epochs-1, len(train_loss_truncated))
    x_valid = np.linspace(0, total_epochs-1, total_epochs)

    if skip_iterations > 0:
        train_loss_truncated = np.array(train_loss_truncated)[x_train>=skip_iterations]
        train_auc_truncated = np.array(train_auc_truncated)[x_train>=skip_iterations]
        x_train = x_train[x_train>=skip_iterations]
        
        valid_loss = np.array(valid_loss)[x_valid>=skip_iterations]
        valid_auc = np.array(valid_auc)[x_valid>=skip_iterations]
        x_valid = x_valid[x_valid>=skip_iterations]

    plt.figure(figsize=(15,8))

    plt.plot(x_train, train_loss_truncated, c='k', label='Training loss')
    plt.plot(x_valid, valid_loss, c='r', linestyle='--', label='Validation loss')

    plt.legend(loc='best')
    plt.show()
    
    plt.figure(figsize=(15,8))

    plt.plot(x_train, train_auc_truncated, c='k', label='Training auc')
    plt.plot(x_valid, valid_auc, c='r', linestyle='--', label='Validation auc')

    plt.legend(loc='best')
    plt.show()