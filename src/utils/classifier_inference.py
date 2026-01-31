import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# -------------------------------------------------
# Load classifier (NO deprecated args)
# -------------------------------------------------
def load_classifier(num_classes, device, checkpoint_path):
    model = resnet18(weights=None)  # ✅ no deprecated 'pretrained'
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------
# Predict image (TOP-K supported)
# -------------------------------------------------
def predict_image(image_tensor, model, class_names, device, top_k=1):
    """
    image_tensor: Tensor [3, H, W] in [0,1]
    returns: list of (class_name, confidence)
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    img = transform(image_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=top_k)

    results = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_idxs, top_probs)
    ]

    return results
