import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default='/mnt/lynx1/datasets/',type=str,help='Root directory path of data')
    parser.add_argument('--csv_path',default='',type=str,help='train files')
    parser.add_argument('--train',default='train.csv',type=str,help='train files')
    parser.add_argument('--test', default='test.csv', type=str, help='test files')
    parser.add_argument('--model_name', default='vggss', type=str, help='test files')
    parser.add_argument('--testset', default='vggss', type=str, help='test files')
    
    parser.add_argument('--hp',default=1,type=int)
    parser.add_argument('--aug',default=1,type=int)
    parser.add_argument('--feature',default=1,type=int)
    parser.add_argument('--intra',default=0,type=int)
    parser.add_argument('--aug_intra',default=0,type=int)
    
    parser.add_argument('--pretrain',default=1,type=int)

    parser.add_argument('--tri_map',action='store_true')
    parser.set_defaults(tri_map=True)
    parser.add_argument('--Neg',action='store_true')
    parser.set_defaults(Neg=True)
    parser.add_argument('--epsilon', default=0.65, type=float)
    parser.add_argument('--epsilon2', default=0.4, type=float)
    parser.add_argument('--epochs',default=80,type=int,help='Number of total epochs to run')
    parser.add_argument('--image_size',default=224,type=int,help='Height and width of inputs')
    parser.add_argument('--learning_rate',default=1e-4,type=float,help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--summaries_dir',default='./logs',type=str)
    parser.add_argument('--checkpoint_dir',default='/mnt/bear2/users/gonhy/tpami_checkpoint',type=str)
    parser.add_argument('--normalisation',default='all',type=str)
    parser.add_argument('--model_depth',default=18,type=int,help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--pool',default="avgpool",type=str,help= 'pooling')
    parser.add_argument('--data_aug',action='store_true')
    parser.set_defaults(data_aug=True)
    parser.add_argument('--write-summarys',action='store_true')
    parser.set_defaults(write_summarys=True)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--n_threads',default=10,type=int,help='Number of threads for multi-thread loading')
    
    parser.add_argument('--random_threshold',default=1,type=int,help='Number of threads for multi-thread loading')
    parser.add_argument('--soft_ep',default=1,type=int,help='Number of threads for multi-thread loading')
    
    parser.add_argument('--opt_name',type=str)
    parser.add_argument('--exp_name',default='previs',type=str)
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
    
    return parser.parse_args()
