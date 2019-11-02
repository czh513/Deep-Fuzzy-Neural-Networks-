from train import CIFAR10_TrainingService

def test_training_small():
    ts = CIFAR10_TrainingService(home_dir='.',
                                 num_batches_per_epoch=5, report_interval=1,
                                 train_batch_size=1024)
    ts.build_and_train(vgg_name='VGG4', n_epochs=1)
    assert ts.last_train_acc >= 0.101

if __name__ == "__main__":
    test_training_small()