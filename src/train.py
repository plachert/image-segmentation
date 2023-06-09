from lightning import Trainer
from model.model import SegmentationModel
from model.architectures.unet import UNet
from data.datamodule import VOCDatamodule
from torchvision import transforms

def replace_tensor_value_(tensor, a, b):
    tensor[tensor == a] = b
    return tensor

def main():
    input_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    target_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: replace_tensor_value_(x.squeeze(0).long(), 255, 21)),
        ]
        
    )
    datamodule = VOCDatamodule(input_transform=input_transform, target_transform=target_transform)
    model = SegmentationModel(UNet())
    trainer = Trainer(max_epochs=1, accelerator='gpu')
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()