import os
from ultralytics import YOLO
import yaml

def main():
    print("Starting YOLOv11 License Plate Detection Training...")
    
    # Set up paths
    dataset_path = "My-First-Project-1"
    data_yaml = os.path.join(dataset_path, "data.yaml")
    
    # Verify dataset exists
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset configuration file not found at {data_yaml}")
        return
    
    # Load and verify data.yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Dataset classes: {data_config['names']}")
    print(f"Number of classes: {data_config['nc']}")
    
    # Initialize YOLOv11 model
    # You can use different model sizes: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    model = YOLO('yolo11n.pt')  # Start with nano model for faster training
    
    # Training parameters
    training_params = {
        'data': data_yaml,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
        'save': True,
        'save_period': -1,
        'cache': False,
        'device': '',
        'workers': 8,
        'project': 'runs/detect',
        'name': 'license_plate_train',
        'exist_ok': False,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': False,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True,
    }
    
    print("Starting training with the following parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    # Start training
    try:
        results = model.train(**training_params)
        print("\nTraining completed successfully!")
        print(f"Best model saved to: {model.trainer.best}")
        
        # Evaluate the model
        print("\nEvaluating model on validation set...")
        metrics = model.val()
        
        print(f"\nValidation Results:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        # Export the model to different formats
        print("\nExporting model...")
        model.export(format='onnx')  # Export to ONNX format
        print("Model exported to ONNX format")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return
    
    print("\nTraining pipeline completed!")

if __name__ == "__main__":
    main()
