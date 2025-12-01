# 🛠️ Install necessary packages
!pip install -q torchvision
!pip install -q matplotlib opencv-python

# 📁 Upload image
from google.colab import files
uploaded = files.upload()  # Choose a .jpg/.png X-ray image

# 📦 Imports
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# 🧠 Load ResNet18 model and modify
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.eval()

# 🧼 Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image

# 🔥 Grad-CAM function
def generate_gradcam(model, image_tensor, target_class=None):
    model.eval()
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    final_conv = model.layer4[1].conv2
    forward = final_conv.register_forward_hook(forward_hook)
    backward = final_conv.register_backward_hook(backward_hook)

    output = model(image_tensor)
    if target_class is None:
        target_class = torch.argmax(output, 1).item()

    model.zero_grad()
    output[0, target_class].backward()

    grads = gradients[0]
    fmap = features[0]

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * fmap, dim=1).squeeze()

    cam = torch.clamp(cam, min=0)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (224, 224))

    forward.remove()
    backward.remove()
    return cam

# 🎯 Predict + visualize + save result
def predict_and_visualize(image_path):
    input_tensor, original_image = preprocess_image(image_path)
    cam = generate_gradcam(model, input_tensor)

    img_cv = np.array(original_image.resize((224, 224)))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)

    # Threshold & draw bounding boxes
    threshold = np.uint8(cam * 255)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100:  # Skip tiny boxes
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Predict
    output = model(input_tensor)
    pred = torch.argmax(output, 1).item()
    label = "Fracture" if pred == 1 else "Normal"

    # Display image
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {label}", fontsize=15)
    plt.axis('off')
    plt.show()

    # 💾 Save image
    output_filename = f"output_{os.path.basename(image_path)}"
    cv2.imwrite(output_filename, overlay)
    print(f"✅ Output image saved as: {output_filename}")

# 📸 Run on uploaded image
import os
image_filename = list(uploaded.keys())[0]
predict_and_visualize(image_filename)
