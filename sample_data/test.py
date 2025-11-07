"""
Script to convert 4-channel TIFF to 3-channel TIFF by removing one channel
"""
from skimage import io
import numpy as np
import os

def remove_channel_from_tiff(input_path, output_path, channel_to_remove=3):
    """
    Remove a channel from a multi-channel TIFF image
    
    Parameters:
    -----------
    input_path : str
        Path to input 4-channel TIFF
    output_path : str
        Path to save output 3-channel TIFF
    channel_to_remove : int
        Index of channel to remove (0-indexed)
    """
    print(f"Loading image from: {input_path}")
    
    # Load the image
    img = io.imread(input_path)
    
    print(f"Original image shape: {img.shape}")
    print(f"Original image dtype: {img.dtype}")
    
    # Check if image has 4 channels
    if len(img.shape) != 3 or img.shape[2] != 4:
        print(f"ERROR: Expected 4-channel image, got shape {img.shape}")
        return False
    
    # Create list of channels to keep
    channels_to_keep = [i for i in range(4) if i != channel_to_remove]
    
    print(f"\nRemoving channel {channel_to_remove}")
    print(f"Keeping channels: {channels_to_keep}")
    
    # Extract the channels we want to keep
    img_3ch = img[:, :, channels_to_keep]
    
    print(f"New image shape: {img_3ch.shape}")
    print(f"New image dtype: {img_3ch.dtype}")
    
    # Save the new image
    print(f"\nSaving to: {output_path}")
    io.imsave(output_path, img_3ch, check_contrast=False)
    
    # Verify the saved image
    print("\nVerifying saved image...")
    img_verify = io.imread(output_path)
    print(f"Verified shape: {img_verify.shape}")
    print(f"Verified dtype: {img_verify.dtype}")
    
    # Calculate file sizes
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\nFile sizes:")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  New:      {new_size:.2f} MB")
    print(f"  Reduced:  {original_size - new_size:.2f} MB ({(1 - new_size/original_size)*100:.1f}%)")
    
    print("\n✓ Channel removal complete!")
    return True


if __name__ == "__main__":
    # Input and output paths
    input_file = "sample_test_image_small.tif"
    output_file = "sample_test_image_small_3ch.tif"
    
    # Which channel to remove (0-indexed)
    # Channel 3 is typically the T-cell channel (CD3)
    channel_to_remove = 3
    
    print("=" * 60)
    print("4-Channel to 3-Channel TIFF Converter")
    print("=" * 60)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Removing channel: {channel_to_remove}")
    print()
    
    # Check if input exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Make sure you're in the sample_data directory")
        exit(1)
    
    # Remove the channel
    success = remove_channel_from_tiff(input_file, output_file, channel_to_remove)
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"\nYou can now use {output_file} for testing the 3-channel pipeline")
        print("\nChannel mapping for the new 3-channel image:")
        print("  Channel 0: DAPI")
        print("  Channel 1: QKI")
        print("  Channel 2: IBA1")
    else:
        print("\nERROR: Failed to remove channel")
        exit(1)