from datasets.video_dataset import VideoDataset

# download with https://github.com/cvdfoundation/kinetics-dataset
class KineticsDataset(VideoDataset):
    def __init__(self, root:str, split:str, n_frames:int, transforms, stride:int=2, transpose=True, n_views=4):
        super().__init__(root=root, split=split, n_frames=n_frames, 
                        transforms=transforms, stride=stride, transpose=transpose, n_views=n_views)