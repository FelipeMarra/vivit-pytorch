##########################
# Base Video Dataset
##########################

import os
import math
import torch
import torchaudio
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root:str, split:str, n_frames:int, transforms, stride:int=2, transpose=True, n_views=4):
        super().__init__()
        self.root = os.path.join(root, split)
        self.split = split
        self.stride = stride
        self.n_frames = n_frames
        self.transforms = transforms
        self.transpose = transpose
        self.n_views = n_views

        self.videos_paths, self.videos_classes, self.idx2class = self.get_videos_paths_and_classes()

    def __getitem__(self, index):
        if self.split == "test":
            return self.get_video_tensor_inference(index, self.n_views)
        else:
            return self.get_video_tensor_non_inference(index)

    def __len__(self):
        return len(self.videos_paths)

    def get_videos_paths_and_classes(self):
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

    def get_video_tensor_non_inference(self, index):
        """
            Returns a dict:
                video: self.n_frames video chunk with temporal stride of self.stride
                class: video class index. Use self.idx2class to get the class name
                path: video path
            With one random view
        """
        video_path = self.videos_paths[index]
        video_class = self.videos_classes[index]

        stride = self.stride
        not_enough_frames = False
        failed = True
        last_failed_video = ''
        while failed:
            try:
                streamer = torchaudio.io.StreamReader(video_path)
                info = streamer.get_src_stream_info(0)

                streamer.add_basic_video_stream(
                    frames_per_chunk = self.n_frames,
                    # Change the frame rate to drop frames (temporal stride). No interpolation is performed.
                    frame_rate= math.ceil(info.frame_rate/stride)
                )

                random_start = self.get_random_start(info, stride)

                if random_start < 0:
                    not_enough_frames = True
                    raise ValueError("Not enought frames in video")

                streamer.seek(random_start)

                # Tensor[T, C, H, W]
                chunk = next(iter(streamer.stream()))[0]

                failed = False
            except:
                if not_enough_frames and last_failed_video != video_path:
                    print("Not enough frames in video:", video_path, "Trying with half of the stride")
                    stride = self.stride // 2
                    not_enough_frames = False
                    last_failed_video = video_path
                else:
                    stride = self.stride
                    print("FAILED on loading video:", video_path)
                    # in case of currupted video get another randomly
                    index = torch.randint(0, len(self.videos_paths), (1,)).item()
                    video_path = self.videos_paths[index]
                    video_class = self.videos_classes[index]

        T, C, H, W = chunk.shape
        if T < self.n_frames:
            pad_size = self.n_frames-T
            pad = torch.zeros((pad_size, C, H, W), dtype=chunk.dtype)
            chunk = torch.cat((chunk, pad), 0)
            print(f"Padded with {pad_size} frames for video {video_path}")
        elif T > self.n_frames:
            chunk = chunk[:self.n_frames, :, :, :]

        chunk = self.transforms(chunk)

        # Conv 3D expecs float Tensor[C, T, H, W]
        if self.transpose:
            chunk = chunk.transpose(0, 1)

        return {
            'video': chunk, 
            'class': video_class, 
            'path': video_path
        }

    def get_video_tensor_inference(self, index, n_views=4):
        """
            Returns a dict:
                video: self.n_frames video chunk with temporal stride of self.stride
                class: video class index. Use self.idx2class to get the class name
                path: video path
            With one clip for each linear spaced view
        """
        video_path = self.videos_paths[index]
        video_class = self.videos_classes[index]

        stride = self.stride
        not_enough_frames = False
        failed = True
        last_failed_video = ''
        while failed:
            try:
                streamer = torchaudio.io.StreamReader(video_path)
                info = streamer.get_src_stream_info(0)

                streamer.add_basic_video_stream(
                    frames_per_chunk = self.n_frames,
                    # Change the frame rate to drop frames (temporal stride). No interpolation is performed.
                    frame_rate= math.ceil(info.frame_rate/stride)
                )

                views = self.get_views_start(info, stride, n_views)

                chunks = []
                for view in views:
                    streamer.seek(view)

                    # Tensor[T, C, H, W]
                    chunk = next(iter(streamer.stream()))[0]

                    T, C, H, W = chunk.shape
                    if T < self.n_frames:
                        pad_size = self.n_frames-T
                        pad = torch.zeros((pad_size, C, H, W), dtype=chunk.dtype)
                        chunk = torch.cat((chunk, pad), 0)
                        print(f"Padded with {pad_size} frames for video {video_path}")
                    elif T > self.n_frames:
                        chunk = chunk[:self.n_frames, :, :, :]

                    chunk:torch.Tensor = self.transforms(chunk)

                    # Conv 3D expecs float Tensor[C, T, H, W]
                    if self.transpose:
                        chunk = chunk.transpose(0, 1).float()

                    chunks.append(chunk)

                failed = False
            except:
                if not_enough_frames and last_failed_video != video_path:
                    print("Not enough frames in video:", video_path, "Trying with half of the stride")
                    stride = self.stride // 2
                    not_enough_frames = False
                    last_failed_video = video_path
                else:
                    stride = self.stride
                    print("FAILED on loading video:", video_path)
                    # in case of currupted video get another randomly
                    index = torch.randint(0, len(self.videos_paths), (1,)).item()
                    video_path = self.videos_paths[index]
                    video_class = self.videos_classes[index]

        chunks = torch.stack(chunks, 0)

        return {
            'views': chunks, 
            'class': video_class, 
            'path': video_path
        }
    
    def get_max_second(self, info, stride):
        """
            Returns the maximum time, in seconds, in the video that still grants that if we start
            from it we'll be able to sample n_frames*stride frames. That's because we need at 
            least n_frames*stride/frame_rate seconds to sample n_frames with the specified stride
        """
        seconds_needed = math.ceil((self.n_frames * stride)/ info.frame_rate)
        duration = math.floor(info.num_frames/info.frame_rate)

        return int(duration - seconds_needed)

    def get_random_start(self, info, stride):
        """
            Returns a random start point in the video constrained to self.get_max_second()
        """
        # Chose random start restricted to our seconds_needed
        max_second = self.get_max_second(info, stride)

        random_start = 0
        if max_second > 0:
            random_start = torch.randint(0, max_second, (1,)).item()
        if max_second < 0:
            random_start = -1

        return random_start

    def get_views_start(self, info, stride, n_views):
        """
            Retunrs the beggining of each view
            Views are linearly spaced across the video duration, but constrained to self.get_max_second()
        """
        max_second = self.get_max_second(info, stride)
        views = torch.linspace(0, max_second, n_views).tolist()
        #print(views)
        return views