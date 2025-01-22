#%%
import os
from datasets.video_dataset import VideoDataset

class GamesDataset(VideoDataset):
    def __init__(self, root:str, split:str, n_frames:int, transforms, stride:int=2, transpose=True, n_views=4):
        super().__init__(root=root, split=split, n_frames=n_frames, 
                        transforms=transforms, stride=stride, transpose=transpose, n_views=n_views)

    def get_videos_paths_and_classes(self):
        classes = sorted(os.listdir(self.root))

        videos_paths = []
        videos_classes = []
        idx2class = {}

        for action_class_idx, action_class in enumerate(classes):
            idx2class[action_class_idx] = action_class
            class_root = os.path.join(self.root, action_class, 'videos')
            class_videos = os.listdir(class_root)

            for class_video_idx, class_video in enumerate(class_videos):
                class_videos[class_video_idx] = os.path.join(class_root, class_video)

            videos_paths = videos_paths + class_videos
            videos_classes = videos_classes + [action_class_idx] * len(class_videos)

        return videos_paths, videos_classes, idx2class