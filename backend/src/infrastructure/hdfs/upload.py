"""
HDFS Upload Script via WebHDFS API

Uploads nonviolence events from analytics CSV to HDFS.
Filters CSV to only include nonviolence events before uploading.
"""

import os
import pandas as pd
import requests
from pathlib import Path


def filter_nonviolence_events(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include nonviolence events."""
    nonviolence_df = df[df['label'] == 'nonviolence']
    print(f"Filtered {len(nonviolence_df)} nonviolence events from {len(df)} total")
    return nonviolence_df


def upload_to_hdfs(file_content: bytes, hdfs_path: str, namenode_url: str = "http://localhost:9870") -> bool:
    """
    Upload data to HDFS using WebHDFS API.
    
    Args:
        file_content: Bytes to upload
        hdfs_path: Target path in HDFS (e.g., /analytics/events.csv)
        namenode_url: WebHDFS namenode URL
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directory
        parent_dir = os.path.dirname(hdfs_path)
        mkdir_url = f"{namenode_url}/webhdfs/v1{parent_dir}"
        mkdir_params = {"op": "MKDIRS", "permission": "755"}
        
        print(f"Creating HDFS directory: {parent_dir}")
        resp = requests.put(mkdir_url, params=mkdir_params)
        if resp.status_code not in [200, 201, 409]:
            print(f"Warning: Directory creation returned {resp.status_code}")
        
        # Upload file via WebHDFS
        webhdfs_url = f"{namenode_url}/webhdfs/v1{hdfs_path}"
        put_params = {"op": "CREATE", "overwrite": "true", "permission": "644"}
        
        print(f"Uploading to HDFS: {hdfs_path}")
        
        # First request gets redirect to datanode
        resp = requests.put(webhdfs_url, params=put_params, allow_redirects=False)
        
        if resp.status_code == 307:
            # Follow redirect to datanode for actual data upload
            redirect_url = resp.headers.get("Location")
            if redirect_url:
                resp = requests.put(redirect_url, data=file_content)
                if resp.status_code == 201:
                    print(f"Successfully uploaded to HDFS: {hdfs_path}")
                    return True
        
        print(f"Upload failed: {resp.status_code}")
        return False
        
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        print(f"Is HDFS running at {namenode_url}?")
        return False
    except Exception as e:
        print(f"Error uploading to HDFS: {e}")
        return False


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent
    input_csv = project_root / "ai_service" / "insights" / "data" / "analytics_events.csv"
    
    if not input_csv.exists():
        print(f"Error: Input CSV not found at {input_csv}")
        return
    
    print("=" * 60)
    print("HDFS Upload - Nonviolence Events")
    print("=" * 60)
    
    try:
        df = pd.read_csv(input_csv)
        nonviolence_df = filter_nonviolence_events(df)
        
        csv_bytes = nonviolence_df.to_csv(index=False).encode('utf-8')
        
        print("\n2. Uploading to HDFS...")
        hdfs_path = "/analytics/nonviolence_events.csv"
        
        success = upload_to_hdfs(csv_bytes, hdfs_path)
        
        if success:
            print(f"\nView in HDFS Web UI: http://localhost:9870")
        else:
            print("\nUpload failed")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
