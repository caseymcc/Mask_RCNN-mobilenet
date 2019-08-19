import os
import sys
import time
import numpy as np
import lprDataset as lpr
import skimage.draw

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class LprConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "lpr"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + license plate

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class LprDataset(utils.Dataset):
    def __init__(self):
        self.lprDataSet=lpr.WG_LprDataset()
        super().__init__()

    def load_annotations(self, dataset_dir, subset):
        # Add classes. We have only one class to add.
        self.add_class("wg_lpr", 1, "licensePlate")

        # Train or validation dataset?
        assert subset in ["train", "val"]
#        dataset_dir = os.path.join(dataset_dir, subset)

        if not self.lprDataSet.directoryOpen:
            self.lprDataSet.openDirectory(dataset_dir)
            self.lprDataSet.loadAllImages()

        imageCount=len(self.lprDataSet.annotatedImages)
        startIndex=0
        endIndex=imageCount-1

        if subset == 'train':
            endIndex=int(imageCount*0.8)
        else:
            startIndex=int(imageCount*0.8)+1

        for annotatedImage in self.lprDataSet.annotatedImages[startIndex:endIndex]:
            image_path=annotatedImage.imageFilePath
            image=skimage.io.imread(annotatedImage.imageFilePath)
            height,width=image.shape[:2]

            polygons=[]
            for annotation in annotatedImage.annotations:
                if annotation.type != lpr.AnnotationType.LicensePlate:
                    continue

                polygons.append(annotation.bbox)
            self.add_image(
                "wg_lpr",
                image_id=annotatedImage.id,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)
        
                
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "wg_lpr":
            return super().load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, polygon in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(polygon.pointsY, polygon.pointsX)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "wg_lpr":
            return info["path"]
        else:
            super().image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = LprDataset()
    dataset_train.load_annotations(args.dataset, "train")
    dataset_train.prepare()
    # Validation dataset
    dataset_val = LprDataset()
    dataset_val.load_annotations(args.dataset, "val")
    dataset_val.prepare()
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

def evaluate(model, image_path=None, video_path=None):
    print("Not implemented yet")


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect license plates.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the Lpr dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to run detection on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to detection on')
    parser.add_argument('--backbone', required=False,
                        default="resnet50",
                        metavar="<backbone>",
                        help='Feature Pyramid Network backbone type')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "evaluate":
        assert args.image or args.video,\
               "Provide --image or --video to evaluate"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Backbone: ", args.backbone)

    # Configurations
    if args.command == "train":
        config = LprConfig()
    else:
        class InferenceConfig(LprConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()

    utils.configure_backbone(config, args.backbone)

    config.display()


    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "evaluate":
        evaluate(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))