"""
HDFS Upload Script

Uploads nonviolence events from analytics CSV to HDFS.
This script filters the CSV to only include nonviolence events
before uploading to HDFS.

Usage:
    python hdfs_upload.py
    
Prerequisites:
    - HDFS namenode and datanode must be running
    - docker compose up hdfs-namenode hdfs-datanode
"""

import os
import pandas as pd
import subprocess
import tempfile
from pathlib import Path


def filter_nonviolence_events(input_csv: str, output_csv: str) -> int:
    """
    Filter CSV to only include nonviolence events.
    
    Args:
        input_csv: Path to input CSV with mixed events
        output_csv: Path to output CSV with only nonviolence events
        
    Returns:
        Number of nonviolence events
    """
    df = pd.read_csv(input_csv)
    
    # Filter only nonviolence events
    nonviolence_df = df[df['label'] == 'nonviolence']
    
    # Save to output file
    nonviolence_df.to_csv(output_csv, index=False)
    
    print(f"Filtered {len(nonviolence_df)} nonviolence events from {len(df)} total events")
    return len(nonviolence_df)


def upload_to_hdfs(local_path: str, hdfs_path: str) -> bool:
    """
    Upload file to HDFS using docker exec.
    
    Args:
        local_path: Path to local file
        hdfs_path: Target path in HDFS
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # First, copy file into the namenode container
        container_path = f"/tmp/{os.path.basename(local_path)}"
        
        # Copy file to container
        copy_cmd = [
            "docker", "cp", 
            local_path, 
            f"hdfs-namenode:{container_path}"
        ]
        print(f"Copying file to container: {' '.join(copy_cmd)}")
        subprocess.run(copy_cmd, check=True)
        
        # Create HDFS directory if it doesn't exist
        mkdir_cmd = [
            "docker", "exec", "hdfs-namenode",
            "hdfs", "dfs", "-mkdir", "-p", os.path.dirname(hdfs_path)
        ]
        print(f"Creating HDFS directory: {' '.join(mkdir_cmd)}")
        subprocess.run(mkdir_cmd, check=False)  # May fail if already exists
        
        # Upload to HDFS
        put_cmd = [
            "docker", "exec", "hdfs-namenode",
            "hdfs", "dfs", "-put", "-f", container_path, hdfs_path
        ]
        print(f"Uploading to HDFS: {' '.join(put_cmd)}")
        subprocess.run(put_cmd, check=True)
        
        # Verify upload
        ls_cmd = [
            "docker", "exec", "hdfs-namenode",
            "hdfs", "dfs", "-ls", hdfs_path
        ]
        result = subprocess.run(ls_cmd, capture_output=True, text=True)
        print(f"HDFS file info: {result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error uploading to HDFS: {e}")
        return False


def main():
    print("=" * 60)
    print("HDFS Upload Script - Nonviolence Events")
    print("=" * 60)
    
    # Paths
    script_dir = Path(__file__).parent
    input_csv = script_dir / "data" / "analytics_events.csv"
    
    if not input_csv.exists():
        print(f"Error: Input CSV not found at {input_csv}")
        print("Please run generate_analytics_dataset.py first")
        return
    
    # Create temp file for filtered data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        output_csv = tmp.name
    
    try:
        # Filter nonviolence events
        print("\n1. Filtering nonviolence events...")
        n_events = filter_nonviolence_events(str(input_csv), output_csv)
        print(f"   Saved {n_events} nonviolence events to temp file")
        
        # Upload to HDFS
        print("\n2. Uploading to HDFS...")
        hdfs_path = "/analytics/nonviolence_events.csv"
        success = upload_to_hdfs(output_csv, hdfs_path)
        
        if success:
            print(f"\n✅ Successfully uploaded to HDFS: {hdfs_path}")
            print(f"   View in HDFS Web UI: http://localhost:9870")
        else:
            print("\n❌ Failed to upload to HDFS")
            
    finally:
        # Cleanup temp file
        if os.path.exists(output_csv):
            os.remove(output_csv)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
