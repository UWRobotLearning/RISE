import torch


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dict = torch.load('/home/kevin/ood/sim_pipeline/sim_pipeline/logs/discriminator_threading_test/20250127215826/models/test.pth')
    import ipdb; ipdb.set_trace()
    
if __name__ == '__main__':
    run()