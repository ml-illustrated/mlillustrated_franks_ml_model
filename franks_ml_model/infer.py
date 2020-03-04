
import os, json, inspect
from PIL import Image
import torch
from torchvision import transforms

from . import EfficientNet

class EfficientNetInfer(object):
    def __init__( self, architecture_name='efficientnet-b0' ):
        super().__init__()

        self.effnet_model = EfficientNet.from_pretrained(architecture_name)
        self.effnet_model.eval()

        self.img_size = EfficientNet.get_image_size(architecture_name)

        self.tfms = transforms.Compose(
            [transforms.Resize(self.img_size),
             transforms.ToTensor(),
             transforms.Normalize(
                 [0.485, 0.456, 0.406],
                 [0.229, 0.224, 0.225]),
            ])

        self.labels_map = self.load_imgnet_labels_map()

    def load_imgnet_labels_map( self ):
        # Load ImageNet class names
        path = os.path.join( os.path.dirname(inspect.getfile(EfficientNetInfer)), 'labels_map.txt' )

        labels_map = json.load(open(path))
        labels_map = [labels_map[str(i)] for i in range(1000)]
        return labels_map

    def load_and_transform_image( self, fn_image ):
        img_tensor = self.tfms(Image.open( fn_image )).unsqueeze(0)
        # print(img.shape) # torch.Size([1, 3, 224, 224])
        return img_tensor

    def infer_batch_image_tensor( self, batch_image_tensor, topk=5 ):
        with torch.no_grad():
            outputs = self.effnet_model( batch_image_tensor )
        
        top_predictions = []
        for idx in torch.topk(outputs, k=topk).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            top_predictions.append( [ idx, self.labels_map[idx], prob ] )

        return top_predictions

    def infer_image( self, fn_image, topk=5 ):
        batch_image_tensor = self.load_and_transform_image( fn_image )
        return self.infer_batch_image_tensor( batch_image_tensor, topk=topk )


if __name__ == '__main__':
    import sys
    
    fn_image = sys.argv[1]

    model = EfficientNetInfer('efficientnet-b0')
    top_predictions = model.infer_image( fn_image )

    for row in top_predictions:
        print( row )
        # print('{label:<75} ({p:.2f}%)'.format(label=row[1], p=row[2]*100))

# pip install -e .
# python demo_effnet/infer.py tests/test_files/dog.jpg
