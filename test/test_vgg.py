from train import CIFAR10_TrainingService

def test_training_small():
    ts = CIFAR10_TrainingService(home_dir='.', normalize_data=True, 
                                 num_batches_per_epoch=5, report_interval=1,
                                 train_batch_size=1024)
    last_train_acc, last_test_acc = ts.build_and_train(vgg_name='VGG4', n_epochs=1)
    assert last_train_acc >= 0.101

if __name__ == "__main__":
    test_training_small()