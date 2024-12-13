#%%
import os
import math
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms import v2

#%%
# download with https://github.com/cvdfoundation/kinetics-dataset
class KineticsDataset(Dataset):
    def __init__(self, root:str, split:str, n_frames:int, transforms, crop_size=224, stride:int=2):
        super().__init__()
        self.root = os.path.join(root, split)
        self.stride = stride
        self.n_frames = n_frames
        self.transforms = transforms
        self.crop_size = crop_size

        self.videos_paths, self.videos_classes, self.idx2class = self.get_videos()

    def __getitem__(self, index):
        """
            Returns a dict:
                video: self.n_frames video chunk with temporal stride of self.stride
                class: video class index. Use self.idx2class to get the class name
                path: video path
        """
        video_path = self.videos_paths[index]
        video_class = self.videos_classes[index]

        failed = True
        while failed:
            try:
                streamer = torchaudio.io.StreamReader(video_path)
                info = streamer.get_src_stream_info(0)

                streamer.add_basic_video_stream(
                    frames_per_chunk = self.n_frames,
                    # Change the frame rate to drop frames (temporal stride of self.stride). No interpolation is performed.
                    frame_rate= math.ceil(info.frame_rate/self.stride)
                )
                random_start = self.get_random_start(info)

                if random_start < 0:
                    print("FAILED on getting a random start. Not enough frames in video:", video_path)

                streamer.seek(random_start)

                # Tensor[T, C, H, W]
                chunk = next(iter(streamer.stream()))[0]

                failed = False
            except:
                print("FAILED on loading video:", video_path)
                # in case of currupted video get another randomly
                index = torch.randint(0, len(self.videos_paths), (1,)).item()
                video_path = self.videos_paths[index]
                video_class = self.videos_classes[index]

        T = chunk.shape[0]
        if T < self.n_frames:
            raise ValueError(f"KineticsDataset getitem function returned {T} frames for n_frames of {self.n_frames} for video {video_path}")
        elif T > self.n_frames:
            chunk = chunk[:self.n_frames, :, :, :]

        chunk = self.transforms(chunk)

        # Conv 3D expecs float Tensor[C, T, H, W]
        chunk = chunk.transpose(0, 1).float()
        return {
                'video': chunk, 
                'class': video_class, 
                'path': video_path
                }

    def __len__(self):
        return len(self.videos_paths)

    def get_videos(self):
        classes = sorted(os.listdir(self.root))

        videos_paths = []
        videos_classes = []
        idx2class = {}

        for action_class_idx, action_class in enumerate(classes):
            idx2class[action_class_idx] = action_class
            class_root = os.path.join(self.root, action_class)
            class_videos = os.listdir(class_root)

            for class_video_idx, class_video in enumerate(class_videos):
                class_videos[class_video_idx] = os.path.join(class_root, class_video)

            videos_paths = videos_paths + class_videos
            videos_classes = videos_classes + [action_class_idx] * len(class_videos)

        return videos_paths, videos_classes, idx2class

    def get_random_start(self, info):
        """
            Returns the maximum time, in seconds, in the video that still grants that if we start
            from it we'll be able to sample n_frames*self.stride frames. That's because we need at 
            least n_frames*self.stride/frame_rate seconds to sample n_frames with stride self.stride
        """
        seconds_needed = math.ceil((self.n_frames * self.stride)/ info.frame_rate)
        duration = math.floor(info.num_frames/info.frame_rate)

        # Chose random start restricted to our seconds_needed
        max_second = int(duration - seconds_needed)

        random_start = 0
        if max_second > 0:
            random_start = torch.randint(0, max_second, (1,)).item()
        if max_second < 0:
            random_start = -1

        return random_start