import torch
import torchvision.transforms as transforms
from.bagging import BaggingClassifier
from.bagging import name_model
name_model = ['tf_efficientnet_b0_ns','tf_efficientnet_b0_ns','tf_efficientnet_b1_ns']
class classifier:
    def __init__(self, model_path, input_size = 128, num_classes = 3, padding = True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_estimators = 3
        self.model = BaggingClassifier(estimator=None,
                                        n_estimators=self.n_estimators,
                                        output_dim=3,
                                        lr = 0.0004,
                                        epochs = 50,
                                        weight_decay = 5e-4)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.input_size = input_size
        self.num_classes = num_classes
        self.padding = padding

        transform = []
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        if padding:
            transform.append(
                lambda x: transforms.Pad(((128 - x.shape[2]) // 2, (128 - x.shape[1]) // 2), fill=0,
                                                     padding_mode="constant")(x))
        transform.append(transforms.Resize((input_size, input_size)))
        self.transform = transforms.Compose(transform)
        print(self.transform)

    def predict(self, input_):
        len_input = len(input_)
        patch = torch.empty(len_input, 3, self.input_size, self.input_size)
        for n, img in enumerate(input_):
            new_img = self.transform(img)
            patch[n] = new_img

        patch = patch.to(self.device)
        with torch.no_grad():
            _, output = torch.max(self.model(patch), -1)
        return output
