from torchvision.datasets import Kinetics

# download with https://github.com/cvdfoundation/kinetics-dataset

kinetics = Kinetics(root='/media/felipe/C0CE73B2CE739EFA/kinetics-dataset/k400', 
                    num_classes='400', 
                    download=False, 
                    frames_per_clip=128, 
                    step_between_clips=2, 
                    num_download_workers=6)
