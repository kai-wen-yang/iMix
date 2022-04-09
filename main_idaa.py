import argparse
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import make_data_loader, mixup_data
import os
import copy
from vae import *
from torch.multiprocessing import set_sharing_strategy
import torchvision
import wandb
set_sharing_strategy("file_system")


def gen_adv(model, vae, x_i, criterion, args):
    x_i = x_i.detach()
    bsz = x_i.size(0)
    with torch.no_grad():
        z, gx, _, _ = vae(x_i)
        z_i = model(x_i, adv=True)

    variable_bottle = Variable(z.detach(), requires_grad=True)
    adv_gx = vae(variable_bottle, True)
    x_j_adv = adv_gx + (x_i - gx).detach()
    z_j_adv = model(x_j_adv, adv=True)

    logits = torch.div(torch.matmul(z_i.detach(), z_j_adv.t()), 0.2)  # Contrastive temp
    tmp_loss = criterion(logits, torch.arange(bsz).cuda())
    tmp_loss.backward()

    with torch.no_grad():
        sign_grad = variable_bottle.grad.data.sign()
        variable_bottle.data = variable_bottle.data + args.eps * sign_grad
        adv_gx = vae(variable_bottle, True)
        x_j_adv = adv_gx + (x_i - gx).detach()
    x_j_adv.requires_grad = False
    x_j_adv.detach()
    return x_j_adv, gx


def reconst_images(x_i, gx, x_j_adv, run):
    grid_X = torchvision.utils.make_grid(x_i[:16].data, nrow=8, padding=2, normalize=True)
    run.log({"X.jpg": [wandb.Image(grid_X)]}, commit=False)
    grid_GX = torchvision.utils.make_grid(gx[:16].data, nrow=8, padding=2, normalize=True)
    run.log({"GX.jpg": [wandb.Image(grid_GX)]}, commit=False)
    grid_RX = torchvision.utils.make_grid((x_i[:16] - gx[:16]).data, nrow=8, padding=2, normalize=True)
    run.log({"RX.jpg": [wandb.Image(grid_RX)]}, commit=False)
    grid_AdvX = torchvision.utils.make_grid(x_j_adv[:16].data, nrow=8, padding=2, normalize=True)
    run.log({"AdvX.jpg": [wandb.Image(grid_AdvX)]}, commit=False)
    grid_delta = torchvision.utils.make_grid((x_j_adv - x_i)[:16].data, nrow=8, padding=2, normalize=True)
    run.log({"Delta.jpg": [wandb.Image(grid_delta)]}, commit=False)
    run.log({'l2_norm': torch.mean((x_j_adv - x_i).reshape(x_i.shape[0], -1).norm(dim=1)),
               'linf_norm': torch.mean((x_j_adv - x_i).reshape(x_i.shape[0], -1).abs().max(dim=1)[0])
               }, commit=False)


class Trainer(object):
    def __init__(self, args):
        self.args = args

        if args.net == "resnet18":
            from nets.resnet import ResNet18
            model = ResNet18(self.args.proj_size, bn_adv_flag=True,
                                                 bn_adv_momentum=args.bn_adv_momentum)
        elif args.net == "resnet50":
            from nets.resnet import ResNet50
            model = ResNet50(self.args.proj_size, bn_adv_flag=True,
                                                 bn_adv_momentum=args.bn_adv_momentum)
        elif args.net == "wideresnet282":
            from nets.wideresnet import WRN28_2
            model = WRN28_2(self.args.proj_size, bn_adv_flag=True,
                                                 bn_adv_momentum=args.bn_adv_momentum)
        else:
            raise NotImplementedError
        vae = CVAE_cifar_withbn(128, args.dim)
        vae.load_state_dict(torch.load(args.vae_path))
        vae.cuda()
        vae.eval()
        print("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        self.model = nn.DataParallel(model).cuda()
        self.vae = vae

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.steps, gamma=self.args.gamma)

        self.criterion = nn.CrossEntropyLoss()
        
        self.kwargs = {"num_workers": 12, "pin_memory": False}        
        self.train_loader, self.val_loader = make_data_loader(args, **self.kwargs)

        self.best = 0
        self.best_epoch = 0
        self.acc = []
        self.train_acc = []
        self.wandb = wandb.init(config=args)

        
    def train(self, epoch):
        running_loss = 0.0
        self.model.train()
        
        acc = 0
        tbar = tqdm(self.train_loader)
        m_dists = torch.tensor([])
        l = torch.tensor([])
        self.epoch = epoch
        total_sum = 0

        tbar.set_description("Training iMix, train_loss {}".format(""))
        
        #iMix + N-pairs
        for i, sample in enumerate(tbar):
            img1, img2 = sample["image1"].cuda(), sample["image2"].cuda()
            bsz = img1.shape[0]
            labels = torch.zeros(len(img1))
            img1, _, _, lam, mix_index = mixup_data(img1, labels, 1.)

            img_adv, gx = gen_adv(self.model, self.vae, img2, self.criterion, self.args)
            self.optimizer.zero_grad()
            z_i = self.model(img1)
            z_j = self.model(img2)
            z_adv = self.model(img_adv, adv=True)
                    
            logits = torch.div(torch.matmul(z_i, z_j.t()), 0.2) #Contrastive temp
            logits_adv = torch.div(torch.matmul(z_i, z_adv.t()), 0.2) #Contrastive temp
            loss = lam * self.criterion(logits, torch.arange(bsz).cuda()) + (1 - lam) * self.criterion(logits, mix_index.cuda())+ \
                   lam * self.criterion(logits_adv, torch.arange(bsz).cuda()) + (1 - lam) * self.criterion(logits_adv, mix_index.cuda())
            if i % 5 == 0:
                tbar.set_description("Training iMix, train loss {:.2f}, lr {:.3f}".format(loss.item(), self.optimizer.param_groups[0]['lr']))
            # compute gradient and do SGD step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        reconst_images(img2, gx, img_adv, self.wandb)
        self.scheduler.step()
        print("Epoch: {0}".format(epoch))
        torch.save({'best':self.best, 'epoch':self.epoch, 'net':self.model.state_dict()}, os.path.join(self.args.save_dir, "last_model.pth.tar"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.args.save_dir, "last_optimizer.pth.tar"))

    def get_train_features(self):
        self.model.eval()
        self.train_feats = torch.zeros((len(self.train_loader.dataset), self.args.proj_size))

        if self.args.dataset == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif self.args.dataset == 'cifar100':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif self.args.dataset == 'miniimagenet':
            mean = [0.4728, 0.4487, 0.4031]
            std = [0.2744, 0.2663 , 0.2806]

        size = 32
        if self.args.dataset == 'miniimagenet':
            size = 84
        original_transforms = copy.deepcopy(self.train_loader.dataset.transform)
        self.train_loader.dataset.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.CenterCrop(size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])
        with torch.no_grad():
            tbar = tqdm(self.train_loader)
            tbar.set_description("Computing features on the train set")
            for i, sample in enumerate(tbar):
                images, index = sample["image1"].cuda(), sample["index"]
                features = self.model(images)
                self.train_feats[index] = features.detach().cpu()
                
        self.train_loader.dataset.transform = original_transforms
        
    def kNN(self, K=200, sigma=0.1):    
        # set the model to evaluation mode
        self.model.eval()
            
        # tracking variables
        total = 0

        self.get_train_features()
        
        trainFeatures = self.train_feats.cuda()
        trainLabels = torch.LongTensor(self.train_loader.dataset.targets).cuda()
        
        trainFeatures = trainFeatures.t()
        C = trainLabels.max() + 1
        C = C.item()
        # start to evaluate                                                                                                                                                                                                                                                     
        top1 = 0.
        top5 = 0.
        tbar = tqdm(self.val_loader)
        tbar.set_description("kNN eval")
        with torch.no_grad():
            retrieval_one_hot = torch.zeros(K, C).cuda()
            for i, sample in enumerate(tbar):
                images, targets = sample["image"].cuda(), sample["target"].cuda()
                batchSize = images.size(0)    
                # forward                                                                                                                                                                                                                                                       
                features = self.model(images)
                
                # cosine similarity                                                                                                                                                                                                                                             
                dist = torch.mm(features, trainFeatures)
                
                yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
                candidates = trainLabels.view(1,-1).expand(batchSize, -1)
                retrieval = torch.gather(candidates, 1, yi)
                
                retrieval_one_hot.resize_(batchSize * K, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(sigma).exp_()
                probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C),
                                            yd_transform.view(batchSize, -1, 1)), 1)
                _, predictions = probs.sort(1, True)
                
                # Find which predictions match the target                                                                                                                                                                                                                             
                correct = predictions.eq(targets.data.view(-1,1))
                
                top1 = top1 + correct.narrow(1,0,1).sum().item()
                top5 = top5 + correct.narrow(1,0,5).sum().item()
                
                total += targets.size(0)
                
        print("kNN accuracy", top1/total)
        if self.best <= top1/total:
            self.best = top1/total
            self.best_epoch = self.epoch
            torch.save({'best':self.best, 'epoch':self.epoch, 'net':self.model.state_dict()}, os.path.join(self.args.save_dir, "best_model.pth.tar"))
            torch.save(self.optimizer.state_dict(), os.path.join(self.args.save_dir, "best_optimizer.pth.tar"))
            
        return top1/total
    
def main():


    parser = argparse.ArgumentParser(description="iMix")
    parser.add_argument("--net", type=str, default="wideresnet282",
                        choices=["resnet18", "wideresnet282", "resnet50"],
                        help="net name")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "miniimagenet"])
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument('--steps', type=int, default=[2000,3000], nargs='+', help='Epochs when to reduce lr')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1, help="Multiplicative factor for lr decrease, default .1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="No cuda")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--proj-size", default=128, type=int)
    parser.add_argument("--no-eval", default=False, action='store_true')

    parser.add_argument('--dim', default=512, type=int, help='CNN_embed_dim')
    parser.add_argument('--adv', default=False, action='store_true', help='adversarial exmaple')
    parser.add_argument('--eps', default=0.01, type=float, help='eps for adversarial')
    parser.add_argument('--bn_adv_momentum', default=0.01, type=float, help='batch norm momentum for advprop')
    parser.add_argument('--vae_path', default='../results/vae_dim512_kl0.1_simclr/model_epoch92.pth',
                        type=str, help='vae_path')
    args = parser.parse_args()
    #For reproducibility purposes
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        
    args.cuda = not args.no_cuda
    
    torch.manual_seed(args.seed)
    
    _trainer = Trainer(args)

    start_ep = 0 

    if args.resume is not None:
        l = torch.load(args.resume)
        start_ep = l['epoch']
        _trainer.best = l['best']
        _trainer.best_epoch = l['epoch']
        _trainer.model.load_state_dict(l['net'])
        _trainer.optimizer.load_state_dict(torch.load(args.resume.replace("model","optimizer")))
        for _ in range(start_ep):
            _trainer.scheduler.step()

    for eps in range(start_ep, args.epochs):
        _trainer.train(eps)
        if not args.no_eval:
            _trainer.kNN()

if __name__ == "__main__":
   main()
