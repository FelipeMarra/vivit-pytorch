#%%
import math
import torch
import torchaudio

corrupted_video_path = '/home/felipe/Desktop/k400/videos/train/braiding hair/B6GxQKcL7IY_000213_000223.mp4'
working_video_path = '/home/felipe/Desktop/k400/videos/train/braiding hair/B7vIU2dHer4_000002_000012.mp4'

n_frames = 32

#%%
try:
    streamer = torchaudio.io.StreamReader(corrupted_video_path)
except:
    print("BBBBBBBBBBBBBBBBB")
    streamer = torchaudio.io.StreamReader(working_video_path)

info = streamer.get_src_stream_info(0)

print(info.frame_rate, info.num_frames, info.num_frames/info.frame_rate)

# We need at least n_frames*2/frame_rate seconds
# to sample n_frames with stride 2
seconds_needed = math.ceil((n_frames * 2)/ info.frame_rate)
print('seconds_needed', seconds_needed)

# Chose random start restricted to our seconds_needed
duration = math.floor(info.num_frames/info.frame_rate)
max_second = int(duration - seconds_needed)
max_second = max_second if max_second >= 0 else 0
random_start = torch.randint(0, max_second, (1,)).item()
print('random_start:', random_start)

#%%
streamer.add_basic_video_stream(
    frames_per_chunk=n_frames,
    # Change the frame rate to drop frames (temporal stride of 2). No interpolation is performed.
    frame_rate= math.ceil(info.frame_rate/2)
)

streamer.seek(random_start)
chunk = next(iter(streamer.stream()))[0]

#%%
print(chunk.shape)
