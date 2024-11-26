from torchvision.datasets import Kinetics

kinetics = Kinetics(root='/media/felipe/C0CE73B2CE739EFA/Kinetics', 
                    num_classes='400', 
                    download=True, 
                    frames_per_clip=128, 
                    step_between_clips=2, 
                    num_download_workers=1)
