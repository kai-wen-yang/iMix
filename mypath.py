class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cifar10':
            return './cifar/'
        elif dataset == 'cifar100':
            return './cifar/'
        elif dataset == 'miniimagenet':
            return './mini/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
        
