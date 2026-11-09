import os
from dotenv import load_dotenv

load_dotenv()


import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url



class ObjectStorage:
    def __init__(self):
        cloudinary.config(secure=True)
        self.folder = os.getenv("CLOUDINARY_FOLDER", "video_frames")

    def upload_jpeg(self, data: bytes, key: str) -> str:
        public_id = key.rsplit(".", 1)[0]

        response = cloudinary.uploader.upload(
            data,
            folder=self.folder,
            public_id=public_id,
            resource_type="image",
            overwrite=True,
            invalidate=True,  # clear CDN cache when re-running the same video_id
        )
        return response["secure_url"]

    def gcs_uri(self, key: str) -> str:
        """Included to match your class interface; returns Cloudinary's secure URL format."""
        public_id = key.rsplit(".", 1)[0] if "." in key else key
        full_public_id = f"{self.folder}/{public_id}" if self.folder else public_id

        # Generate a optimized/secure delivery URL
        url, _ = cloudinary_url(full_public_id, format="jpg", secure=True)
        return url