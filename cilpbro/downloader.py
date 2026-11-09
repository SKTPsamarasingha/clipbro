import os
import yt_dlp
import imageio_ffmpeg

from cilpbro import progress


def download_video(url, output_folder="video"):
    # Create the folder automatically if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save inside the folder using the video ID as the filename
    output_template = os.path.join(output_folder, "%(id)s.%(ext)s")

    ydl_opts = {
        'format': 'bv*[height<=480]+ba/b',
        'outtmpl': output_template,
        'overwrites': True,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [progress.ytdlp_hook()],
        'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe(),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Get the exact file path that was saved (e.g., 'video\dQw4w9WgXcQ.mp4')
        downloaded_path = ydl.prepare_filename(info)
        final_path = os.path.splitext(downloaded_path)[0] + ".mp4"

    return final_path
