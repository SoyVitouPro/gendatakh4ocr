#!/usr/bin/env python3
import os
import json
from pathlib import Path

def create_metadata_json():
    base_dir = Path("/data/Vitou/gen_data/image_generated")
    metadata = []
    
    annotation_dirs = [
        "image_0_2000",
        "image_2000_4000", 
        "image_4000_6000",
        "image_6000_8000",
        "image_8000_10000",
        "image_10000_12000",
        "image_12000_14000",
        "image_14000_16000", 
        "image_16000_18000",
        "image_18000_20000"
    ]
    
    for dir_name in annotation_dirs:
        annotation_file = base_dir / dir_name / "annotations.txt"
        
        if annotation_file.exists():
            print(f"Processing {annotation_file}")
            
            with open(annotation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            image_path = parts[0]
                            label = parts[1]
                            
                            full_path = str(base_dir.parent / image_path)
                            
                            metadata.append({
                                "image_path": full_path,
                                "label": label,
                                "directory": dir_name
                            })
    
    output_file = base_dir.parent / "meta_data.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Created {output_file} with {len(metadata)} entries")
    return len(metadata)

if __name__ == "__main__":
    count = create_metadata_json()
    print(f"Successfully merged annotations from all directories into meta_data.json")
    print(f"Total entries: {count}")